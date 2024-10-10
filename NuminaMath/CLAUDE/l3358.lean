import Mathlib

namespace lars_bakeshop_production_l3358_335821

/-- Lars' bakeshop productivity calculation -/
theorem lars_bakeshop_production :
  let loaves_per_hour : ℕ := 10
  let baguettes_per_two_hours : ℕ := 30
  let working_hours_per_day : ℕ := 6
  
  let loaves_per_day : ℕ := loaves_per_hour * working_hours_per_day
  let baguette_intervals : ℕ := working_hours_per_day / 2
  let baguettes_per_day : ℕ := baguettes_per_two_hours * baguette_intervals
  
  loaves_per_day + baguettes_per_day = 105 :=
by
  sorry

end lars_bakeshop_production_l3358_335821


namespace porter_painting_sale_l3358_335806

/-- The sale price of Porter's previous painting in dollars -/
def previous_sale : ℕ := 9000

/-- The sale price of Porter's most recent painting in dollars -/
def recent_sale : ℕ := 5 * previous_sale - 1000

theorem porter_painting_sale : recent_sale = 44000 := by
  sorry

end porter_painting_sale_l3358_335806


namespace quadratic_max_value_l3358_335855

theorem quadratic_max_value (m : ℝ) (h_m : m ≠ 0) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, m * x^2 - 2 * m * x + 2 ≤ 4) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, m * x^2 - 2 * m * x + 2 = 4) →
  m = 2/3 ∨ m = -2 := by
sorry

end quadratic_max_value_l3358_335855


namespace no_valid_triples_l3358_335822

theorem no_valid_triples :
  ¬∃ (x y z : ℕ+), 
    (Nat.lcm x.val y.val = 180) ∧
    (Nat.lcm x.val z.val = 450) ∧
    (Nat.lcm y.val z.val = 600) ∧
    (x.val + y.val + z.val = 120) := by
  sorry

end no_valid_triples_l3358_335822


namespace product_of_sines_equality_l3358_335847

theorem product_of_sines_equality : 
  (1 + Real.sin (π/12)) * (1 + Real.sin (5*π/12)) * (1 + Real.sin (7*π/12)) * (1 + Real.sin (11*π/12)) = 
  (1 + Real.sin (π/12))^2 * (1 + Real.sin (5*π/12))^2 := by
sorry

end product_of_sines_equality_l3358_335847


namespace meal_cost_theorem_l3358_335882

theorem meal_cost_theorem (initial_people : ℕ) (additional_people : ℕ) (share_decrease : ℚ) :
  initial_people = 5 →
  additional_people = 3 →
  share_decrease = 15 →
  let total_people := initial_people + additional_people
  let total_cost := (initial_people * share_decrease * total_people) / (total_people - initial_people)
  total_cost = 200 :=
by sorry

end meal_cost_theorem_l3358_335882


namespace intersection_of_A_and_B_l3358_335849

-- Define set A
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by sorry

end intersection_of_A_and_B_l3358_335849


namespace expression_evaluation_l3358_335884

theorem expression_evaluation (x : ℤ) (h1 : -1 ≤ x) (h2 : x ≤ 2) 
  (h3 : x ≠ 1) (h4 : x ≠ 0) (h5 : x ≠ 2) : 
  (x^2 - 1) / (x^2 - 2*x + 1) + (x^2 - 2*x) / (x - 2) / x = 1 := by
  sorry

end expression_evaluation_l3358_335884


namespace shaded_perimeter_equals_48_l3358_335807

/-- Represents a circle in the arrangement -/
structure Circle where
  circumference : ℝ

/-- Represents the arrangement of four circles -/
structure CircleArrangement where
  circles : Fin 4 → Circle
  symmetric : Bool
  touching : Bool

/-- Calculates the perimeter of the shaded region -/
def shadedPerimeter (arrangement : CircleArrangement) : ℝ :=
  sorry

theorem shaded_perimeter_equals_48 (arrangement : CircleArrangement) 
    (h1 : ∀ i, (arrangement.circles i).circumference = 48) 
    (h2 : arrangement.symmetric = true) 
    (h3 : arrangement.touching = true) : 
  shadedPerimeter arrangement = 48 := by
  sorry

end shaded_perimeter_equals_48_l3358_335807


namespace triangle_side_length_l3358_335880

theorem triangle_side_length (a c b : ℝ) (B : ℝ) : 
  a = 3 * Real.sqrt 3 → 
  c = 2 → 
  B = 150 * π / 180 → 
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B → 
  b = 7 := by
sorry

end triangle_side_length_l3358_335880


namespace leak_drain_time_l3358_335805

/-- Given a pump that can fill a tank in 2 hours without a leak,
    and takes 2 1/7 hours to fill the tank with a leak,
    the time it takes for the leak to drain the entire tank is 30 hours. -/
theorem leak_drain_time (fill_time_no_leak fill_time_with_leak : ℚ) : 
  fill_time_no_leak = 2 →
  fill_time_with_leak = 2 + 1 / 7 →
  (1 / (1 / fill_time_no_leak - 1 / fill_time_with_leak)) = 30 := by
  sorry

end leak_drain_time_l3358_335805


namespace min_colors_is_thirteen_l3358_335874

/-- A coloring function for a 25x25 chessboard. -/
def Coloring := Fin 25 → Fin 25 → ℕ

/-- Predicate to check if a coloring satisfies the given condition. -/
def ValidColoring (c : Coloring) : Prop :=
  ∀ i j s t, 1 ≤ i ∧ i < j ∧ j ≤ 25 ∧ 1 ≤ s ∧ s < t ∧ t ≤ 25 →
    (c i s ≠ c j s ∨ c i s ≠ c j t ∨ c j s ≠ c j t)

/-- The minimum number of colors needed for a valid coloring. -/
def MinColors : ℕ := 13

/-- Theorem stating that 13 is the smallest number of colors needed for a valid coloring. -/
theorem min_colors_is_thirteen :
  (∃ c : Coloring, ValidColoring c ∧ (∀ i j, c i j < MinColors)) ∧
  (∀ n : ℕ, n < MinColors →
    ¬∃ c : Coloring, ValidColoring c ∧ (∀ i j, c i j < n)) := by
  sorry

end min_colors_is_thirteen_l3358_335874


namespace quadratic_equation_solutions_l3358_335897

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := fun x => x^2 - 2*x - 8
  ∃ (x₁ x₂ : ℝ), x₁ = 4 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end quadratic_equation_solutions_l3358_335897


namespace fanfan_distance_is_120_l3358_335892

/-- Represents the cost and distance information for a shared car journey -/
structure JourneyInfo where
  ningning_cost : ℝ
  leilei_cost : ℝ
  fanfan_cost : ℝ
  ningning_distance : ℝ

/-- Calculates the distance to Fanfan's home given the journey information -/
def calculate_fanfan_distance (info : JourneyInfo) : ℝ :=
  sorry

/-- Theorem stating that given the journey information, Fanfan's home is 120 km from school -/
theorem fanfan_distance_is_120 (info : JourneyInfo) 
  (h1 : info.ningning_cost = 10)
  (h2 : info.leilei_cost = 25)
  (h3 : info.fanfan_cost = 85)
  (h4 : info.ningning_distance = 12) :
  calculate_fanfan_distance info = 120 := by
  sorry

end fanfan_distance_is_120_l3358_335892


namespace simplify_expression_l3358_335859

theorem simplify_expression (b : ℝ) : ((3 * b + 6) - 6 * b) / 3 = -b + 2 := by
  sorry

end simplify_expression_l3358_335859


namespace product_of_roots_l3358_335895

theorem product_of_roots (x : ℝ) : (x - 4) * (x + 5) = -24 → ∃ y : ℝ, (x * y = 4 ∧ (y - 4) * (y + 5) = -24) := by
  sorry

end product_of_roots_l3358_335895


namespace sqrt_square_abs_l3358_335858

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by
  sorry

end sqrt_square_abs_l3358_335858


namespace gcd_lcm_product_24_60_l3358_335862

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end gcd_lcm_product_24_60_l3358_335862


namespace coat_price_calculation_l3358_335815

/-- The total selling price of a coat after discount and tax -/
def totalSellingPrice (originalPrice discount taxRate : ℝ) : ℝ :=
  let salePrice := originalPrice * (1 - discount)
  salePrice * (1 + taxRate)

/-- Theorem: The total selling price of a $120 coat with 30% discount and 8% tax is $90.72 -/
theorem coat_price_calculation :
  totalSellingPrice 120 0.3 0.08 = 90.72 := by
  sorry

end coat_price_calculation_l3358_335815


namespace max_of_min_is_sqrt_two_l3358_335804

theorem max_of_min_is_sqrt_two (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (⨅ z ∈ ({x, 1/y, y + 1/x} : Set ℝ), z) ≤ Real.sqrt 2 ∧
  ∃ x y, x > 0 ∧ y > 0 ∧ (⨅ z ∈ ({x, 1/y, y + 1/x} : Set ℝ), z) = Real.sqrt 2 := by
sorry

end max_of_min_is_sqrt_two_l3358_335804


namespace find_divisor_l3358_335891

theorem find_divisor (x n : ℕ) (h1 : ∃ k : ℕ, x = k * n + 27)
                     (h2 : ∃ m : ℕ, x = 8 * m + 3)
                     (h3 : n > 27) :
  n = 32 := by
sorry

end find_divisor_l3358_335891


namespace bath_frequency_l3358_335853

/-- 
Given a person who takes a bath B times per week and a shower once per week,
prove that if they clean themselves 156 times in 52 weeks, then B = 2.
-/
theorem bath_frequency (B : ℕ) 
  (h1 : B + 1 = (156 : ℕ) / 52) : B = 2 := by
  sorry

#check bath_frequency

end bath_frequency_l3358_335853


namespace odd_function_property_l3358_335814

/-- A function f(x) = ax^5 - bx^3 + cx is odd and f(-3) = 7 implies f(3) = -7 -/
theorem odd_function_property (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^5 - b * x^3 + c * x)
  (h2 : f (-3) = 7) : 
  f 3 = -7 := by
  sorry

end odd_function_property_l3358_335814


namespace double_price_profit_percentage_l3358_335877

theorem double_price_profit_percentage (cost : ℝ) (initial_profit_rate : ℝ) 
  (initial_selling_price : ℝ) (new_selling_price : ℝ) (new_profit_rate : ℝ) :
  initial_profit_rate = 0.20 →
  initial_selling_price = cost * (1 + initial_profit_rate) →
  new_selling_price = 2 * initial_selling_price →
  new_profit_rate = (new_selling_price - cost) / cost →
  new_profit_rate = 1.40 :=
by sorry

end double_price_profit_percentage_l3358_335877


namespace area_of_PQRS_l3358_335835

/-- A circle with an inscribed square ABCD and another inscribed square PQRS -/
structure InscribedSquares where
  /-- The radius of the circle -/
  r : ℝ
  /-- The side length of square ABCD -/
  s : ℝ
  /-- Half the side length of square PQRS -/
  t : ℝ
  /-- The area of square ABCD is 4 -/
  h_area : s^2 = 4
  /-- The radius of the circle is related to the side of ABCD -/
  h_radius : r^2 = 2 * s^2
  /-- Relationship between r, s, and t based on the Pythagorean theorem -/
  h_pythagorean : (s/2 + t)^2 + t^2 = r^2

/-- The area of square PQRS in the configuration of InscribedSquares -/
def areaOfPQRS (cfg : InscribedSquares) : ℝ := (2 * cfg.t)^2

/-- Theorem stating that the area of PQRS is 2 - √3 -/
theorem area_of_PQRS (cfg : InscribedSquares) : areaOfPQRS cfg = 2 - Real.sqrt 3 := by
  sorry

end area_of_PQRS_l3358_335835


namespace point_to_line_distance_l3358_335883

theorem point_to_line_distance (a : ℝ) : 
  (∃ d : ℝ, d = 4 ∧ d = |3*a - 4*6 - 2| / Real.sqrt (3^2 + (-4)^2)) →
  (a = 2 ∨ a = 46/3) :=
sorry

end point_to_line_distance_l3358_335883


namespace arrangements_equal_l3358_335896

/-- The number of arrangements when adding 2 books to 3 existing books while keeping their relative order --/
def arrangements_books : ℕ := 20

/-- The number of arrangements for 7 people with height constraints --/
def arrangements_people : ℕ := 20

/-- Theorem stating that both arrangement problems result in 20 different arrangements --/
theorem arrangements_equal : arrangements_books = arrangements_people := by
  sorry

end arrangements_equal_l3358_335896


namespace final_result_l3358_335863

theorem final_result (chosen_number : ℕ) (h : chosen_number = 1152) : 
  (chosen_number / 6 : ℚ) - 189 = 3 := by
  sorry

end final_result_l3358_335863


namespace max_sum_given_sum_of_squares_and_product_l3358_335841

theorem max_sum_given_sum_of_squares_and_product (x y : ℝ) :
  x^2 + y^2 = 100 → xy = 40 → x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end max_sum_given_sum_of_squares_and_product_l3358_335841


namespace six_balls_three_boxes_l3358_335829

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 77 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 77 := by
  sorry

end six_balls_three_boxes_l3358_335829


namespace inverse_f_at_486_l3358_335801

/-- Given a function f with the properties f(5) = 2 and f(3x) = 3f(x) for all x,
    prove that the inverse function f⁻¹ evaluated at 486 is equal to 1215. -/
theorem inverse_f_at_486 (f : ℝ → ℝ) (h1 : f 5 = 2) (h2 : ∀ x, f (3 * x) = 3 * f x) :
  Function.invFun f 486 = 1215 := by
  sorry

end inverse_f_at_486_l3358_335801


namespace sin_pi_sixth_plus_tan_pi_third_l3358_335879

theorem sin_pi_sixth_plus_tan_pi_third :
  Real.sin (π / 6) + Real.tan (π / 3) = 1 / 2 + Real.sqrt 3 := by
  sorry

end sin_pi_sixth_plus_tan_pi_third_l3358_335879


namespace floor_equality_iff_range_l3358_335861

theorem floor_equality_iff_range (x : ℝ) : 
  ⌊⌊3 * x⌋ - 1⌋ = ⌊x + 1⌋ ↔ 2/3 ≤ x ∧ x < 4/3 :=
by sorry

end floor_equality_iff_range_l3358_335861


namespace inequality_solution_range_l3358_335865

/-- Given an inequality (ax-1)(x+1) < 0 with respect to x, where the solution set
    is (-∞, 1/a) ∪ (-1, +∞), prove that the range of the real number a is -1 ≤ a < 0. -/
theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, (a * x - 1) * (x + 1) < 0 ↔ x ∈ ({y : ℝ | y < (1 : ℝ) / a} ∪ {y : ℝ | y > -1})) →
  -1 ≤ a ∧ a < 0 :=
sorry

end inequality_solution_range_l3358_335865


namespace height_percentage_difference_l3358_335833

theorem height_percentage_difference (A B : ℝ) (h : B = A * (1 + 1/3)) :
  (B - A) / B = 1/4 := by sorry

end height_percentage_difference_l3358_335833


namespace multiple_of_119_l3358_335809

theorem multiple_of_119 : ∃ k : ℤ, 119 = 7 * k ∧ 
  (∀ m : ℤ, 119 ≠ 2 * m) ∧ 
  (∀ n : ℤ, 119 ≠ 3 * n) ∧ 
  (∀ p : ℤ, 119 ≠ 5 * p) ∧ 
  (∀ q : ℤ, 119 ≠ 11 * q) := by
  sorry

end multiple_of_119_l3358_335809


namespace repeating_decimal_prime_l3358_335898

/-- A function that determines if a rational number has a repeating decimal representation with a given period length. -/
def has_repeating_decimal_period (q : ℚ) (period : ℕ) : Prop :=
  ∃ (k : ℕ) (r : ℚ), q = k + r ∧ r < 1 ∧ (10 ^ period : ℚ) * r = r

theorem repeating_decimal_prime (n : ℕ) (h1 : n > 1) 
  (h2 : has_repeating_decimal_period (1 / n : ℚ) (n - 1)) : 
  Nat.Prime n :=
sorry

end repeating_decimal_prime_l3358_335898


namespace inequality_solution_l3358_335826

theorem inequality_solution (x : ℝ) : 
  x + |2*x + 3| ≥ 2 ↔ x ≤ -5 ∨ x ≥ -1/3 := by sorry

end inequality_solution_l3358_335826


namespace place_value_sum_place_value_sum_holds_l3358_335873

theorem place_value_sum : Real → Prop :=
  fun x => 
    let ten_thousands : Real := 4
    let thousands : Real := 3
    let hundreds : Real := 7
    let tens : Real := 5
    let ones : Real := 2
    let tenths : Real := 8
    let hundredths : Real := 4
    x = ten_thousands * 10000 + thousands * 1000 + hundreds * 100 + 
        tens * 10 + ones + tenths / 10 + hundredths / 100 ∧ 
    x = 43752.84

theorem place_value_sum_holds : ∃ x, place_value_sum x := by
  sorry

end place_value_sum_place_value_sum_holds_l3358_335873


namespace leakage_empty_time_l3358_335860

/-- Given a pipe that fills a tank in 'a' hours without leakage,
    and takes 7 times longer with leakage, the time taken by the leakage
    alone to empty the tank is (7a/6) hours. -/
theorem leakage_empty_time (a : ℝ) (h : a > 0) :
  let fill_time_with_leakage := 7 * a
  let fill_rate := 1 / a
  let combined_fill_rate := 1 / fill_time_with_leakage
  let leakage_rate := fill_rate - combined_fill_rate
  leakage_rate⁻¹ = 7 * a / 6 :=
by sorry

end leakage_empty_time_l3358_335860


namespace arcsin_one_half_eq_pi_sixth_l3358_335802

theorem arcsin_one_half_eq_pi_sixth : Real.arcsin (1/2) = π/6 := by
  sorry

end arcsin_one_half_eq_pi_sixth_l3358_335802


namespace jungkook_balls_count_l3358_335899

/-- The number of boxes Jungkook has -/
def num_boxes : ℕ := 3

/-- The number of balls in each box -/
def balls_per_box : ℕ := 2

/-- The total number of balls Jungkook has -/
def total_balls : ℕ := num_boxes * balls_per_box

theorem jungkook_balls_count : total_balls = 6 := by
  sorry

end jungkook_balls_count_l3358_335899


namespace eel_length_problem_l3358_335889

theorem eel_length_problem (jenna_eel : ℝ) (bill_eel : ℝ) : 
  jenna_eel = (1/3 : ℝ) * bill_eel → 
  jenna_eel + bill_eel = 64 → 
  jenna_eel = 16 := by
  sorry

end eel_length_problem_l3358_335889


namespace max_sum_product_l3358_335846

theorem max_sum_product (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_eq_200 : a + b + c + d = 200) : 
  a * b + b * c + c * d + d * a ≤ 10000 := by
sorry

end max_sum_product_l3358_335846


namespace strawberry_jelly_sales_l3358_335878

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ
  apricot : ℕ
  mixedFruit : ℕ

/-- Conditions for jelly sales -/
def validJellySales (s : JellySales) : Prop :=
  s.grape = 4 * s.strawberry ∧
  s.raspberry = 2 * s.plum ∧
  s.apricot = s.grape / 2 ∧
  s.mixedFruit = 3 * s.raspberry ∧
  s.raspberry = s.grape / 3 ∧
  s.plum = 8

theorem strawberry_jelly_sales (s : JellySales) (h : validJellySales s) : s.strawberry = 12 := by
  sorry

end strawberry_jelly_sales_l3358_335878


namespace smallest_number_divisible_after_subtraction_l3358_335844

theorem smallest_number_divisible_after_subtraction : ∃ (n : ℕ), 
  (∀ (m : ℕ), m > 0 → (m - 8 : ℤ) % 9 = 0 ∧ (m - 8 : ℤ) % 6 = 0 ∧ 
   (m - 8 : ℤ) % 12 = 0 ∧ (m - 8 : ℤ) % 18 = 0 → m ≥ n) ∧
  (n - 8 : ℤ) % 9 = 0 ∧ (n - 8 : ℤ) % 6 = 0 ∧ 
  (n - 8 : ℤ) % 12 = 0 ∧ (n - 8 : ℤ) % 18 = 0 ∧
  n = 44 :=
by sorry

end smallest_number_divisible_after_subtraction_l3358_335844


namespace multiplier_problem_l3358_335869

theorem multiplier_problem (n : ℝ) (h : n = 15) : 
  ∃ m : ℝ, 2 * n = (26 - n) + 19 ∧ n * m = 2 * n := by
  sorry

end multiplier_problem_l3358_335869


namespace mothers_day_bouquet_l3358_335871

/-- Represents the flower shop problem --/
structure FlowerShop where
  carnation_price : ℚ
  rose_price : ℚ
  processing_fee : ℚ
  total_budget : ℚ
  total_flowers : ℕ

/-- Represents a bouquet composition --/
structure Bouquet where
  carnations : ℕ
  roses : ℕ

/-- Checks if a bouquet satisfies the conditions of the flower shop problem --/
def is_valid_bouquet (shop : FlowerShop) (bouquet : Bouquet) : Prop :=
  let total_cost := shop.carnation_price * bouquet.carnations + shop.rose_price * bouquet.roses + shop.processing_fee
  bouquet.carnations + bouquet.roses = shop.total_flowers ∧
  total_cost = shop.total_budget

/-- The main theorem to prove --/
theorem mothers_day_bouquet : 
  let shop := FlowerShop.mk 1.5 2 2 21 10
  let bouquet := Bouquet.mk 2 8
  is_valid_bouquet shop bouquet := by
  sorry

end mothers_day_bouquet_l3358_335871


namespace blue_hat_cost_l3358_335832

theorem blue_hat_cost (total_hats : ℕ) (green_hat_cost : ℕ) (total_price : ℕ) (green_hats : ℕ) :
  total_hats = 85 →
  green_hat_cost = 7 →
  total_price = 548 →
  green_hats = 38 →
  (total_price - green_hats * green_hat_cost) / (total_hats - green_hats) = 6 :=
by sorry

end blue_hat_cost_l3358_335832


namespace log_sum_equals_three_l3358_335885

theorem log_sum_equals_three : 
  Real.log 0.125 / Real.log 0.5 + Real.log (Real.log (Real.log 64 / Real.log 4) / Real.log 3) / Real.log 2 = 3 := by
  sorry

end log_sum_equals_three_l3358_335885


namespace computer_purchase_cost_l3358_335893

/-- Calculates the total cost of John's computer purchase --/
theorem computer_purchase_cost (computer_cost : ℝ) (base_video_card_cost : ℝ) 
  (monitor_foreign_cost : ℝ) (exchange_rate : ℝ) :
  computer_cost = 1500 →
  base_video_card_cost = 300 →
  monitor_foreign_cost = 200 →
  exchange_rate = 1.25 →
  ∃ total_cost : ℝ,
    total_cost = 
      (computer_cost + 
       (0.25 * computer_cost) + 
       (2.5 * base_video_card_cost * 0.88) + 
       ((0.25 * computer_cost) * 1.05) - 
       (0.07 * (computer_cost + (0.25 * computer_cost) + (2.5 * base_video_card_cost * 0.88))) + 
       (monitor_foreign_cost / exchange_rate)) ∧
    total_cost = 2536.30 := by
  sorry

end computer_purchase_cost_l3358_335893


namespace smallest_m_divisibility_l3358_335851

def a : ℕ → ℕ
  | 0 => 3
  | n + 1 => (n + 2) * a n - (n + 1)

def divisible (x y : ℕ) : Prop := ∃ k, y = k * x

theorem smallest_m_divisibility :
  ∀ m : ℕ, m ≥ 2005 →
    (divisible (a (m + 1) - 1) (a m ^ 2 - 1) ∧
     ∀ k : ℕ, 2005 ≤ k ∧ k < m →
       ¬divisible (a (k + 1) - 1) (a k ^ 2 - 1)) ↔
    m = 2010 := by sorry

end smallest_m_divisibility_l3358_335851


namespace circle_symmetry_line_l3358_335817

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the xy-plane of the form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Predicate to check if a circle is symmetric with respect to a line -/
def is_symmetric (c : Circle) (l : Line) : Prop :=
  c.center.1 + c.center.2 = l.slope * c.center.1 + l.intercept + c.center.2

theorem circle_symmetry_line (b : ℝ) : 
  let c : Circle := { center := (1, 2), radius := 1 }
  let l : Line := { slope := 1, intercept := b }
  is_symmetric c l → b = 1 := by
  sorry

#check circle_symmetry_line

end circle_symmetry_line_l3358_335817


namespace decimal_to_base5_conversion_l3358_335850

-- Define a function to convert a base-5 number to base-10
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define the base-5 representation of the number we want to prove
def base5Representation : List Nat := [0, 0, 2, 1]

-- State the theorem
theorem decimal_to_base5_conversion :
  base5ToBase10 base5Representation = 175 := by
  sorry

end decimal_to_base5_conversion_l3358_335850


namespace smallest_k_for_distinct_roots_l3358_335812

theorem smallest_k_for_distinct_roots (k : ℤ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 3 * x - 9/4 = 0 ∧ k * y^2 - 3 * y - 9/4 = 0) →
  k ≥ 1 :=
by sorry

end smallest_k_for_distinct_roots_l3358_335812


namespace sqrt_x_plus_sqrt_y_equals_two_l3358_335842

theorem sqrt_x_plus_sqrt_y_equals_two (θ : ℝ) (x y : ℝ) 
  (h1 : x + y = 3 - Real.cos (4 * θ)) 
  (h2 : x - y = 4 * Real.sin (2 * θ)) : 
  Real.sqrt x + Real.sqrt y = 2 := by
  sorry

end sqrt_x_plus_sqrt_y_equals_two_l3358_335842


namespace distance_between_points_l3358_335830

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-5, 2)
  let p2 : ℝ × ℝ := (7, 7)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 13 := by
sorry

end distance_between_points_l3358_335830


namespace round_table_seating_l3358_335824

theorem round_table_seating (W M : ℕ) : 
  W = 19 → 
  M = 16 → 
  (7 : ℕ) + 12 = W → 
  (3 : ℕ) * 12 = 3 * W - 3 * M → 
  W + M = 35 := by
sorry

end round_table_seating_l3358_335824


namespace tangent_line_to_ellipse_l3358_335811

/-- Given a line y = mx + 3 tangent to the ellipse 4x^2 + y^2 = 4, m^2 = 5 -/
theorem tangent_line_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 3 → 4 * x^2 + y^2 = 4) → 
  (∃! x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + y^2 = 4) → 
  m^2 = 5 := by
  sorry

end tangent_line_to_ellipse_l3358_335811


namespace probability_red_green_white_l3358_335866

def red_marbles : ℕ := 4
def green_marbles : ℕ := 5
def white_marbles : ℕ := 11

def total_marbles : ℕ := red_marbles + green_marbles + white_marbles

theorem probability_red_green_white :
  (red_marbles : ℚ) / total_marbles *
  green_marbles / (total_marbles - 1) *
  white_marbles / (total_marbles - 2) = 11 / 342 := by
  sorry

end probability_red_green_white_l3358_335866


namespace volunteer_selection_count_l3358_335867

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of students to be selected -/
def num_selected : ℕ := 4

/-- The total number of students -/
def total_students : ℕ := num_boys + num_girls

/-- The number of ways to select 4 students from 7 students -/
def total_selections : ℕ := Nat.choose total_students num_selected

/-- The number of ways to select 4 boys from 4 boys -/
def all_boys_selections : ℕ := Nat.choose num_boys num_selected

theorem volunteer_selection_count :
  total_selections - all_boys_selections = 34 := by
  sorry

end volunteer_selection_count_l3358_335867


namespace janet_action_figures_l3358_335837

theorem janet_action_figures (x : ℕ) : 
  (x - 2 : ℤ) + 2 * (x - 2 : ℤ) = 24 → x = 10 := by
  sorry

end janet_action_figures_l3358_335837


namespace prob_heads_tails_heads_l3358_335823

/-- A fair coin has equal probability of landing heads or tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of a sequence of independent events is the product of their individual probabilities -/
def prob_independent_events (p q r : ℝ) : ℝ := p * q * r

/-- The probability of getting heads, tails, then heads when flipping a fair coin three times is 1/8 -/
theorem prob_heads_tails_heads :
  ∀ (p : ℝ), fair_coin p →
  prob_independent_events p p p = 1/8 :=
sorry

end prob_heads_tails_heads_l3358_335823


namespace third_packing_number_l3358_335843

theorem third_packing_number (N : ℕ) (h1 : N = 301) (h2 : N % 3 = 1) (h3 : N % 4 = 1) (h4 : N % 7 = 0) :
  ∃ x : ℕ, x ≠ 3 ∧ x ≠ 4 ∧ x > 4 ∧ N % x = 1 ∧ (∀ y : ℕ, y ≠ 3 ∧ y ≠ 4 ∧ y < x ∧ y > 4 → N % y ≠ 1) ∧ x = 6 :=
by sorry

end third_packing_number_l3358_335843


namespace factorization_of_2x_squared_minus_18_l3358_335808

theorem factorization_of_2x_squared_minus_18 (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by
  sorry

end factorization_of_2x_squared_minus_18_l3358_335808


namespace alyssas_allowance_l3358_335840

theorem alyssas_allowance (allowance : ℝ) : 
  (allowance / 2 + 8 = 12) → allowance = 8 := by
  sorry

end alyssas_allowance_l3358_335840


namespace solve_equation_l3358_335834

theorem solve_equation (x : ℝ) : 3 + 2 * (x - 3) = 24.16 → x = 13.58 := by
  sorry

end solve_equation_l3358_335834


namespace smallest_integer_divisibility_l3358_335875

theorem smallest_integer_divisibility (n : ℕ) : 
  ∃ (a_n : ℤ), 
    (a_n > (Real.sqrt 3 + 1)^(2*n)) ∧ 
    (∀ (x : ℤ), x > (Real.sqrt 3 + 1)^(2*n) → a_n ≤ x) ∧ 
    (∃ (k : ℤ), a_n = 2^(n+1) * k) :=
by sorry

end smallest_integer_divisibility_l3358_335875


namespace intersection_determines_a_l3358_335894

theorem intersection_determines_a (a : ℝ) : 
  let A : Set ℝ := {1, 2}
  let B : Set ℝ := {a, a^2 + 3}
  A ∩ B = {1} → a = 1 := by
sorry

end intersection_determines_a_l3358_335894


namespace unique_root_quadratic_l3358_335870

-- Define the geometric sequence
def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ :=
  b * x^2 + c * x + a

-- State the theorem
theorem unique_root_quadratic (a b c : ℝ) :
  is_geometric_sequence a b c →
  a ≤ b →
  b ≤ c →
  c ≤ 1 →
  (∃! x : ℝ, quadratic a b c x = 0) →
  (∃ x : ℝ, quadratic a b c x = 0 ∧ x = -(Real.rpow 4 (1/3) / 2)) :=
by sorry

end unique_root_quadratic_l3358_335870


namespace basketball_team_cutoff_l3358_335813

theorem basketball_team_cutoff (girls boys callback : ℕ) 
  (h1 : girls = 17) 
  (h2 : boys = 32) 
  (h3 : callback = 10) : 
  girls + boys - callback = 39 := by
  sorry

end basketball_team_cutoff_l3358_335813


namespace equation_solution_l3358_335820

theorem equation_solution (x y : ℝ) :
  3 * x^2 - 12 * y^2 = 0 ↔ (x = 2*y ∨ x = -2*y) :=
by sorry

end equation_solution_l3358_335820


namespace unique_n_l3358_335856

theorem unique_n : ∃! n : ℤ, 
  50 ≤ n ∧ n ≤ 150 ∧ 
  7 ∣ n ∧ 
  n % 9 = 3 ∧ 
  n % 4 = 3 ∧
  n = 147 := by
sorry

end unique_n_l3358_335856


namespace article_price_proof_l3358_335845

-- Define the original price
def original_price : ℝ := 2500

-- Define the profit percentage
def profit_percentage : ℝ := 0.25

-- Define the profit amount
def profit_amount : ℝ := 625

-- Theorem statement
theorem article_price_proof :
  profit_amount = original_price * profit_percentage :=
by
  sorry

#check article_price_proof

end article_price_proof_l3358_335845


namespace manufacturer_profit_is_18_percent_l3358_335839

/- Define the given values -/
def customer_price : ℚ := 30.09
def retailer_profit_percentage : ℚ := 25
def wholesaler_profit_percentage : ℚ := 20
def manufacturer_cost : ℚ := 17

/- Define the calculation steps -/
def retailer_cost : ℚ := customer_price / (1 + retailer_profit_percentage / 100)
def wholesaler_price : ℚ := retailer_cost
def wholesaler_cost : ℚ := wholesaler_price / (1 + wholesaler_profit_percentage / 100)
def manufacturer_price : ℚ := wholesaler_cost

/- Define the manufacturer's profit percentage calculation -/
def manufacturer_profit_percentage : ℚ := 
  (manufacturer_price - manufacturer_cost) / manufacturer_cost * 100

/- The theorem to prove -/
theorem manufacturer_profit_is_18_percent : 
  manufacturer_profit_percentage = 18 := by sorry

end manufacturer_profit_is_18_percent_l3358_335839


namespace number_of_pupils_l3358_335818

theorem number_of_pupils (total_people : ℕ) (parents : ℕ) (pupils : ℕ) : 
  total_people = 676 → parents = 22 → pupils = total_people - parents → pupils = 654 := by
  sorry

end number_of_pupils_l3358_335818


namespace repeating_decimal_56_eq_fraction_l3358_335857

/-- The decimal representation of a number with infinitely repeating digits 56 after the decimal point -/
def repeating_decimal_56 : ℚ :=
  56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to 56/99 -/
theorem repeating_decimal_56_eq_fraction : repeating_decimal_56 = 56 / 99 := by
  sorry

end repeating_decimal_56_eq_fraction_l3358_335857


namespace exponential_comparison_l3358_335852

theorem exponential_comparison : 0.3^2.1 < 2.1^0.3 := by
  sorry

end exponential_comparison_l3358_335852


namespace greatest_drop_in_april_l3358_335803

/-- Represents the months from January to June -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June

/-- The price change for each month -/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January  => -1.00
  | Month.February => 2.50
  | Month.March    => 0.00
  | Month.April    => -3.00
  | Month.May      => -1.50
  | Month.June     => 1.00

/-- A month has a price drop if its price change is negative -/
def has_price_drop (m : Month) : Prop :=
  price_change m < 0

/-- The greatest monthly drop in price occurred in April -/
theorem greatest_drop_in_april :
  ∀ m : Month, has_price_drop m → price_change Month.April ≤ price_change m :=
by sorry

end greatest_drop_in_april_l3358_335803


namespace sqrt_sqrt_two_power_ten_l3358_335800

theorem sqrt_sqrt_two_power_ten : (Real.sqrt ((Real.sqrt 2) ^ 4)) ^ 10 = 1024 := by
  sorry

end sqrt_sqrt_two_power_ten_l3358_335800


namespace arithmetic_sequence_count_l3358_335876

/-- Given an arithmetic sequence with first term a₁ = -3, last term aₙ = 45, 
    and common difference d = 3, prove that the number of terms n is 17. -/
theorem arithmetic_sequence_count : 
  ∀ (n : ℕ) (a : ℕ → ℤ), 
    a 1 = -3 ∧ 
    (∀ k, a (k + 1) = a k + 3) ∧ 
    a n = 45 → 
    n = 17 := by
  sorry

end arithmetic_sequence_count_l3358_335876


namespace unique_non_right_triangle_l3358_335836

/-- A function that checks if three numbers can form a right-angled triangle -/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The theorem stating that among the given sets, only (7, 24, 26) cannot form a right-angled triangle -/
theorem unique_non_right_triangle :
  is_right_triangle 3 4 5 ∧
  is_right_triangle 5 12 13 ∧
  is_right_triangle 8 15 17 ∧
  ¬ is_right_triangle 7 24 26 :=
sorry

end unique_non_right_triangle_l3358_335836


namespace equivalence_proof_l3358_335872

variable (P Q : Prop)

theorem equivalence_proof :
  (P → Q) ↔ (¬Q → ¬P) ∧ (¬P ∨ Q) ∧ ¬((Q → P) ↔ (P → Q)) :=
sorry

end equivalence_proof_l3358_335872


namespace g_of_x_plus_3_l3358_335854

def g (x : ℝ) : ℝ := x^2 - x

theorem g_of_x_plus_3 : g (x + 3) = x^2 + 5*x + 6 := by sorry

end g_of_x_plus_3_l3358_335854


namespace inequality_proof_l3358_335868

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ 0) :
  (x^2 * y) / z + (y^2 * z) / x + (z^2 * x) / y ≥ x^2 + y^2 + z^2 := by
  sorry

end inequality_proof_l3358_335868


namespace pure_imaginary_complex_number_l3358_335886

theorem pure_imaginary_complex_number (m : ℝ) : 
  (∃ (z : ℂ), z = (m^2 - 1) + (m - 1) * I ∧ z.re = 0 ∧ z.im ≠ 0) → m = -1 :=
sorry

end pure_imaginary_complex_number_l3358_335886


namespace largest_circle_area_l3358_335827

/-- The area of the largest circle formed from a string with length equal to the perimeter of a 15x9 rectangle is 576/π. -/
theorem largest_circle_area (length width : ℝ) (h1 : length = 15) (h2 : width = 9) :
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * π)
  π * radius^2 = 576 / π :=
by sorry

end largest_circle_area_l3358_335827


namespace remaining_fabric_area_l3358_335828

/-- Calculates the remaining fabric area after cutting curtains -/
theorem remaining_fabric_area (bolt_length bolt_width living_room_length living_room_width bedroom_length bedroom_width : ℝ) 
  (h1 : bolt_length = 16)
  (h2 : bolt_width = 12)
  (h3 : living_room_length = 4)
  (h4 : living_room_width = 6)
  (h5 : bedroom_length = 2)
  (h6 : bedroom_width = 4) :
  bolt_length * bolt_width - (living_room_length * living_room_width + bedroom_length * bedroom_width) = 160 := by
  sorry

#check remaining_fabric_area

end remaining_fabric_area_l3358_335828


namespace sara_quarters_l3358_335864

/-- The total number of quarters after receiving additional quarters -/
def total_quarters (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Sara's total quarters is the sum of her initial quarters and additional quarters -/
theorem sara_quarters : total_quarters 21 49 = 70 := by
  sorry

end sara_quarters_l3358_335864


namespace min_value_and_inequality_l3358_335810

theorem min_value_and_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (∃ (y : ℝ), y = (a + 1/a) * (b + 1/b) ∧ 
    (∀ (z : ℝ), z = (a + 1/a) * (b + 1/b) → y ≤ z) ∧ 
    y = 25/4) ∧ 
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end min_value_and_inequality_l3358_335810


namespace jordan_rectangle_length_l3358_335888

/-- Given two rectangles with equal area, where one rectangle measures 5 inches by 24 inches
    and the other has a width of 15 inches, prove that the length of the second rectangle is 8 inches. -/
theorem jordan_rectangle_length (carol_length carol_width jordan_width : ℕ) 
  (h1 : carol_length = 5)
  (h2 : carol_width = 24)
  (h3 : jordan_width = 15)
  (h4 : carol_length * carol_width = jordan_width * (carol_length * carol_width / jordan_width)) :
  carol_length * carol_width / jordan_width = 8 := by
  sorry

#check jordan_rectangle_length

end jordan_rectangle_length_l3358_335888


namespace five_distinct_values_of_triple_exponentiation_l3358_335819

def exponentiateThree (n : ℕ) : ℕ := 3^n

theorem five_distinct_values_of_triple_exponentiation :
  ∃! (s : Finset ℕ), 
    (∀ x ∈ s, ∃ f : ℕ → ℕ → ℕ, x = f (exponentiateThree 3) (exponentiateThree (exponentiateThree 3))) ∧ 
    s.card = 5 := by
  sorry

end five_distinct_values_of_triple_exponentiation_l3358_335819


namespace travel_time_calculation_l3358_335887

theorem travel_time_calculation (total_distance : ℝ) (foot_speed : ℝ) (bicycle_speed : ℝ) (foot_distance : ℝ)
  (h1 : total_distance = 61)
  (h2 : foot_speed = 4)
  (h3 : bicycle_speed = 9)
  (h4 : foot_distance = 16) :
  (foot_distance / foot_speed) + ((total_distance - foot_distance) / bicycle_speed) = 9 :=
by sorry

end travel_time_calculation_l3358_335887


namespace fraction_zero_at_zero_l3358_335890

theorem fraction_zero_at_zero (x : ℝ) : 
  (2 * x) / (x + 3) = 0 ↔ x = 0 :=
by sorry

end fraction_zero_at_zero_l3358_335890


namespace intersection_A_complement_B_l3358_335848

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt ((6 / (x + 1)) - 1)}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 2*x + 3)}

-- Statement to prove
theorem intersection_A_complement_B : 
  A ∩ (Set.univ \ B) = {x : ℝ | 3 ≤ x ∧ x ≤ 5} := by sorry

end intersection_A_complement_B_l3358_335848


namespace road_repaving_l3358_335881

theorem road_repaving (total_repaved : ℕ) (previously_repaved : ℕ) :
  total_repaved = 4938 ∧ previously_repaved = 4133 →
  total_repaved - previously_repaved = 805 := by
  sorry

end road_repaving_l3358_335881


namespace spade_heart_eval_l3358_335816

/-- Operation ♠ for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- Operation ♥ for real numbers -/
def heart (x y : ℝ) : ℝ := x^2 - y^2

/-- Theorem stating that 5 ♠ (3 ♥ 2) = 0 -/
theorem spade_heart_eval : spade 5 (heart 3 2) = 0 := by
  sorry

end spade_heart_eval_l3358_335816


namespace senior_japanese_fraction_l3358_335825

theorem senior_japanese_fraction (j : ℝ) (s : ℝ) (x : ℝ) :
  s = 2 * j →                     -- Senior class is twice the size of junior class
  (1 / 3) * (j + s) = (3 / 4) * j + x * s →  -- 1/3 of all students equals 3/4 of juniors plus x fraction of seniors
  x = 1 / 8 :=                    -- Fraction of seniors studying Japanese
by sorry

end senior_japanese_fraction_l3358_335825


namespace power_sum_equals_six_l3358_335838

theorem power_sum_equals_six : 2 - 2^2 - 2^3 - 2^4 - 2^5 - 2^6 - 2^7 - 2^8 - 2^9 + 2^10 = 6 := by
  sorry

end power_sum_equals_six_l3358_335838


namespace prime_sum_square_l3358_335831

theorem prime_sum_square (p q r : ℕ) (n : ℕ+) :
  Prime p → Prime q → Prime r → p^(n:ℕ) + q^(n:ℕ) = r^2 → n = 1 := by
sorry

end prime_sum_square_l3358_335831
