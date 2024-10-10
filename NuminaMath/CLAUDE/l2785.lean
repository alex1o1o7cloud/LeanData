import Mathlib

namespace system_of_equations_solution_l2785_278507

theorem system_of_equations_solution
  (a b c x y z : ℝ)
  (h1 : x - a * y + a^2 * z = a^3)
  (h2 : x - b * y + b^2 * z = b^3)
  (h3 : x - c * y + c^2 * z = c^3)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hca : c ≠ a) :
  x = a * b * c ∧ y = a * b + b * c + c * a ∧ z = a + b + c :=
by sorry

end system_of_equations_solution_l2785_278507


namespace p_range_q_range_p_or_q_false_range_l2785_278548

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (1 - 2*m) + y^2 / (m + 3) = 1 ∧ (1 - 2*m) * (m + 3) < 0

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 3 - 2*m = 0

-- Theorem for the range of m where p is true
theorem p_range (m : ℝ) : p m ↔ m < -3 ∨ m > 1/2 := by sorry

-- Theorem for the range of m where q is true
theorem q_range (m : ℝ) : q m ↔ m ≤ -3 ∨ m ≥ 1 := by sorry

-- Theorem for the range of m where "p ∨ q" is false
theorem p_or_q_false_range (m : ℝ) : ¬(p m ∨ q m) ↔ -3 < m ∧ m ≤ 1/2 := by sorry

end p_range_q_range_p_or_q_false_range_l2785_278548


namespace tracy_candies_l2785_278500

theorem tracy_candies (x : ℕ) : x = 68 :=
  -- Initial number of candies
  have h1 : x > 0 := by sorry

  -- After eating 1/4, the remaining candies are divisible by 3 (for giving 1/3 to Rachel)
  have h2 : ∃ k : ℕ, 3 * k = 3 * x / 4 := by sorry

  -- After giving 1/3 to Rachel, the remaining candies are even (for Tracy and mom to eat 12 each)
  have h3 : ∃ m : ℕ, 2 * m = x / 2 := by sorry

  -- After Tracy and mom eat 12 each, the remaining candies are between 7 and 11
  have h4 : 7 ≤ x / 2 - 24 ∧ x / 2 - 24 ≤ 11 := by sorry

  -- Final number of candies is 5
  have h5 : ∃ b : ℕ, 2 ≤ b ∧ b ≤ 6 ∧ x / 2 - 24 - b = 5 := by sorry

  sorry

end tracy_candies_l2785_278500


namespace octal_subtraction_l2785_278526

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Subtraction operation in base 8 --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from decimal to octal --/
def to_octal (n : ℕ) : OctalNumber :=
  sorry

/-- Conversion from octal to decimal --/
def from_octal (n : OctalNumber) : ℕ :=
  sorry

theorem octal_subtraction :
  octal_sub (to_octal 43) (to_octal 22) = to_octal 21 :=
by sorry

end octal_subtraction_l2785_278526


namespace rectangle_fold_area_l2785_278554

theorem rectangle_fold_area (a b : ℝ) (h1 : a = 4) (h2 : b = 8) : 
  let diagonal := Real.sqrt (a^2 + b^2)
  let height := diagonal / 2
  (1/2) * diagonal * height = 10 := by
  sorry

end rectangle_fold_area_l2785_278554


namespace cherry_sales_analysis_l2785_278518

/-- Represents the daily sales of cherries -/
structure CherrySales where
  purchase_price : ℝ
  min_selling_price : ℝ
  max_selling_price : ℝ
  sales_function : ℝ → ℝ
  profit_function : ℝ → ℝ

/-- The specific cherry sales scenario -/
def cherry_scenario : CherrySales where
  purchase_price := 20
  min_selling_price := 20
  max_selling_price := 40
  sales_function := λ x => -2 * x + 160
  profit_function := λ x => (x - 20) * (-2 * x + 160)

theorem cherry_sales_analysis (c : CherrySales) 
  (h1 : c.purchase_price = 20)
  (h2 : c.min_selling_price = 20)
  (h3 : c.max_selling_price = 40)
  (h4 : c.sales_function 25 = 110)
  (h5 : c.sales_function 30 = 100)
  (h6 : ∀ x, c.min_selling_price ≤ x ∧ x ≤ c.max_selling_price → 
    c.sales_function x = -2 * x + 160)
  (h7 : ∀ x, c.profit_function x = (x - c.purchase_price) * (c.sales_function x)) :
  (∀ x, c.sales_function x = -2 * x + 160) ∧ 
  (∃ x, c.min_selling_price ≤ x ∧ x ≤ c.max_selling_price ∧ c.profit_function x = 1000 ∧ x = 30) ∧
  (∃ x, c.min_selling_price ≤ x ∧ x ≤ c.max_selling_price ∧ 
    ∀ y, c.min_selling_price ≤ y ∧ y ≤ c.max_selling_price → c.profit_function x ≥ c.profit_function y) ∧
  (∃ x, c.profit_function x = 1600 ∧ x = 40) := by
  sorry

#check cherry_sales_analysis cherry_scenario

end cherry_sales_analysis_l2785_278518


namespace winnie_lollipops_l2785_278520

/-- The number of lollipops Winnie keeps for herself -/
def lollipops_kept (cherry wintergreen grape shrimp friends : ℕ) : ℕ :=
  (cherry + wintergreen + grape + shrimp) % friends

theorem winnie_lollipops :
  lollipops_kept 36 125 8 241 13 = 7 := by
  sorry

end winnie_lollipops_l2785_278520


namespace ratio_of_segments_l2785_278566

/-- Given collinear points A, B, C in the Cartesian plane where:
    A = (a, 0) lies on the x-axis
    B lies on the line y = x
    C lies on the line y = 2x
    AB/BC = 2
    D = (a, a)
    E is the second intersection of the circumcircle of triangle ADC with y = x
    F is the intersection of ray AE with y = 2x
    Prove that AE/EF = √2/2 -/
theorem ratio_of_segments (a : ℝ) : ∃ (B C E F : ℝ × ℝ),
  let A := (a, 0)
  let D := (a, a)
  -- B lies on y = x
  B.2 = B.1 ∧
  -- C lies on y = 2x
  C.2 = 2 * C.1 ∧
  -- AB/BC = 2
  (((B.1 - A.1)^2 + (B.2 - A.2)^2) : ℝ) / ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 4 ∧
  -- E lies on y = x
  E.2 = E.1 ∧
  -- E is on the circumcircle of ADC
  (E.1 - A.1)^2 + (E.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
  (E.1 - C.1)^2 + (E.2 - C.2)^2 = (D.1 - C.1)^2 + (D.2 - C.2)^2 ∧
  -- F lies on y = 2x
  F.2 = 2 * F.1 ∧
  -- F lies on ray AE
  ∃ (t : ℝ), t > 0 ∧ F.1 - A.1 = t * (E.1 - A.1) ∧ F.2 - A.2 = t * (E.2 - A.2) →
  -- Conclusion: AE/EF = √2/2
  (((E.1 - A.1)^2 + (E.2 - A.2)^2) : ℝ) / ((F.1 - E.1)^2 + (F.2 - E.2)^2) = 1/2 := by
sorry

end ratio_of_segments_l2785_278566


namespace prime_representation_l2785_278547

theorem prime_representation (p : ℕ) (hp : p.Prime) (hp_gt_2 : p > 2) :
  (p % 8 = 1 → ∃ x y : ℤ, ↑p = x^2 + 16 * y^2) ∧
  (p % 8 = 5 → ∃ x y : ℤ, ↑p = 4 * x^2 + 4 * x * y + 5 * y^2) :=
by sorry

end prime_representation_l2785_278547


namespace paco_cookies_l2785_278573

/-- The number of sweet cookies Paco had initially -/
def initial_sweet_cookies : ℕ := 34

/-- The number of salty cookies Paco had initially -/
def initial_salty_cookies : ℕ := 97

/-- The number of sweet cookies Paco ate -/
def eaten_sweet_cookies : ℕ := 15

/-- The number of salty cookies Paco ate -/
def eaten_salty_cookies : ℕ := 56

/-- The number of sweet cookies Paco had left after eating -/
def remaining_sweet_cookies : ℕ := 19

theorem paco_cookies : initial_sweet_cookies = eaten_sweet_cookies + remaining_sweet_cookies :=
by sorry

end paco_cookies_l2785_278573


namespace house_rent_calculation_l2785_278521

def salary : ℚ := 170000

def food_fraction : ℚ := 1/5
def clothes_fraction : ℚ := 3/5
def remaining_amount : ℚ := 17000

def house_rent_fraction : ℚ := 1/10

theorem house_rent_calculation :
  house_rent_fraction * salary + food_fraction * salary + clothes_fraction * salary + remaining_amount = salary :=
by sorry

end house_rent_calculation_l2785_278521


namespace points_per_treasure_l2785_278512

/-- Calculates the points per treasure in Tiffany's video game. -/
theorem points_per_treasure (treasures_level1 treasures_level2 total_score : ℕ) : 
  treasures_level1 = 3 → treasures_level2 = 5 → total_score = 48 →
  total_score / (treasures_level1 + treasures_level2) = 6 := by
  sorry

end points_per_treasure_l2785_278512


namespace quadratic_function_m_value_l2785_278534

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (A B C : ℤ) : ℝ → ℝ := fun x ↦ A * x^2 + B * x + C

theorem quadratic_function_m_value
  (A B C : ℤ)
  (h1 : QuadraticFunction A B C 2 = 0)
  (h2 : 100 < QuadraticFunction A B C 9 ∧ QuadraticFunction A B C 9 < 110)
  (h3 : 150 < QuadraticFunction A B C 10 ∧ QuadraticFunction A B C 10 < 160)
  (h4 : ∃ m : ℤ, 10000 * m < QuadraticFunction A B C 200 ∧ QuadraticFunction A B C 200 < 10000 * (m + 1)) :
  ∃ m : ℤ, m = 16 ∧ 10000 * m < QuadraticFunction A B C 200 ∧ QuadraticFunction A B C 200 < 10000 * (m + 1) :=
sorry

end quadratic_function_m_value_l2785_278534


namespace vector_addition_l2785_278511

theorem vector_addition : 
  let v1 : Fin 3 → ℝ := ![(-5 : ℝ), 1, -4]
  let v2 : Fin 3 → ℝ := ![0, 8, -4]
  v1 + v2 = ![(-5 : ℝ), 9, -8] := by sorry

end vector_addition_l2785_278511


namespace wall_area_2_by_4_l2785_278579

/-- The area of a rectangular wall -/
def wall_area (width height : ℝ) : ℝ := width * height

/-- Theorem: The area of a wall that is 2 feet wide and 4 feet tall is 8 square feet -/
theorem wall_area_2_by_4 : wall_area 2 4 = 8 := by
  sorry

end wall_area_2_by_4_l2785_278579


namespace max_sales_revenue_l2785_278501

/-- Sales volume function -/
def f (t : ℕ) : ℝ := -2 * t + 200

/-- Price function -/
def g (t : ℕ) : ℝ :=
  if t ≤ 30 then 0.5 * t + 30 else 40

/-- Daily sales revenue function -/
def S (t : ℕ) : ℝ := f t * g t

/-- The maximum daily sales revenue occurs at t = 20 and is equal to 6400 -/
theorem max_sales_revenue :
  ∃ (t : ℕ), t ∈ Finset.range 50 ∧
  S t = 6400 ∧
  ∀ (t' : ℕ), t' ∈ Finset.range 50 → S t' ≤ S t :=
by sorry

end max_sales_revenue_l2785_278501


namespace jane_ribbons_per_dress_l2785_278590

/-- The number of ribbons Jane adds to each dress --/
def ribbons_per_dress (dresses_first_week : ℕ) (dresses_second_week : ℕ) (total_ribbons : ℕ) : ℚ :=
  total_ribbons / (dresses_first_week + dresses_second_week)

/-- Theorem stating that Jane adds 2 ribbons to each dress --/
theorem jane_ribbons_per_dress :
  ribbons_per_dress (7 * 2) (2 * 3) 40 = 2 := by
  sorry

end jane_ribbons_per_dress_l2785_278590


namespace bus_average_speed_l2785_278591

/-- Proves that given a bicycle traveling at 15 km/h and a bus starting 195 km behind it,
    if the bus catches up to the bicycle in 3 hours, then the average speed of the bus is 80 km/h. -/
theorem bus_average_speed
  (bicycle_speed : ℝ)
  (initial_distance : ℝ)
  (catch_up_time : ℝ)
  (h1 : bicycle_speed = 15)
  (h2 : initial_distance = 195)
  (h3 : catch_up_time = 3)
  : (initial_distance + bicycle_speed * catch_up_time) / catch_up_time = 80 :=
by
  sorry

end bus_average_speed_l2785_278591


namespace only_cylinder_has_quadrilateral_cross_section_l2785_278531

-- Define the types of solids
inductive Solid
  | Cone
  | Cylinder
  | Sphere

-- Define a function that determines if a solid can have a quadrilateral cross-section
def canHaveQuadrilateralCrossSection (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => True
  | _ => False

-- Theorem statement
theorem only_cylinder_has_quadrilateral_cross_section :
  ∀ s : Solid, canHaveQuadrilateralCrossSection s ↔ s = Solid.Cylinder :=
by
  sorry


end only_cylinder_has_quadrilateral_cross_section_l2785_278531


namespace math_test_problem_l2785_278578

theorem math_test_problem (total_questions word_problems steve_answers difference : ℕ)
  (h1 : total_questions = 45)
  (h2 : word_problems = 17)
  (h3 : steve_answers = 38)
  (h4 : difference = total_questions - steve_answers)
  (h5 : difference = 7) :
  total_questions - word_problems - steve_answers = 21 :=
by sorry

end math_test_problem_l2785_278578


namespace maintenance_interval_doubled_l2785_278596

/-- 
Given an original maintenance check interval and a percentage increase,
this function calculates the new maintenance check interval.
-/
def new_maintenance_interval (original : ℕ) (percent_increase : ℕ) : ℕ :=
  original * (100 + percent_increase) / 100

/-- 
Theorem: If the original maintenance check interval is 30 days and 
the interval is increased by 100%, then the new interval is 60 days.
-/
theorem maintenance_interval_doubled :
  new_maintenance_interval 30 100 = 60 := by
  sorry

end maintenance_interval_doubled_l2785_278596


namespace infinitely_many_powers_of_five_with_consecutive_zeros_l2785_278593

theorem infinitely_many_powers_of_five_with_consecutive_zeros : 
  ∀ k : ℕ, ∃ S : Set ℕ, (Set.Infinite S) ∧ 
  (∀ m ∈ S, (5^m : ℕ) ≡ 1 [MOD 2^k]) ∧
  (∃ N : ℕ, ∀ n ≥ N, ∃ m ∈ S, 
    (∃ i : ℕ, (5^m : ℕ) / 10^i % 10^1976 = 0)) :=
by sorry

end infinitely_many_powers_of_five_with_consecutive_zeros_l2785_278593


namespace tuesday_lost_revenue_l2785_278560

/-- Represents a movie theater with its capacity, ticket price, and tickets sold. -/
structure MovieTheater where
  capacity : ℕ
  ticketPrice : ℚ
  ticketsSold : ℕ

/-- Calculates the lost revenue for a movie theater. -/
def lostRevenue (theater : MovieTheater) : ℚ :=
  (theater.capacity - theater.ticketsSold) * theater.ticketPrice

/-- Theorem stating that the lost revenue for the given theater scenario is $208.00. -/
theorem tuesday_lost_revenue :
  let theater : MovieTheater := ⟨50, 8, 24⟩
  lostRevenue theater = 208 := by sorry

end tuesday_lost_revenue_l2785_278560


namespace smallest_n_congruence_l2785_278598

theorem smallest_n_congruence : ∃! n : ℕ+, 
  (∀ m : ℕ+, 5 * m ≡ 2023 [MOD 26] → n ≤ m) ∧ 
  5 * n ≡ 2023 [MOD 26] := by
  sorry

end smallest_n_congruence_l2785_278598


namespace candies_per_house_l2785_278580

/-- Proves that the number of candies received from each house is 7,
    given that there are 5 houses in a block and 35 candies are received from each block. -/
theorem candies_per_house
  (houses_per_block : ℕ)
  (candies_per_block : ℕ)
  (h1 : houses_per_block = 5)
  (h2 : candies_per_block = 35) :
  candies_per_block / houses_per_block = 7 :=
by sorry

end candies_per_house_l2785_278580


namespace pony_jeans_discount_rate_l2785_278523

theorem pony_jeans_discount_rate :
  let fox_price : ℚ := 15
  let pony_price : ℚ := 18
  let fox_quantity : ℕ := 3
  let pony_quantity : ℕ := 2
  let total_savings : ℚ := 9
  let total_discount_rate : ℚ := 25

  ∀ (fox_discount pony_discount : ℚ),
    fox_discount + pony_discount = total_discount_rate →
    fox_quantity * fox_price * (fox_discount / 100) + 
    pony_quantity * pony_price * (pony_discount / 100) = total_savings →
    pony_discount = 25 :=
by
  sorry

end pony_jeans_discount_rate_l2785_278523


namespace function_value_at_negative_a_l2785_278575

/-- Given a function f(x) = x + 1/x - 2 and a real number a such that f(a) = 3,
    prove that f(-a) = -7. -/
theorem function_value_at_negative_a 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x + 1/x - 2) 
  (h2 : f a = 3) : 
  f (-a) = -7 := by
  sorry

end function_value_at_negative_a_l2785_278575


namespace questionnaire_C_count_l2785_278506

/-- Represents the system sampling method described in the problem -/
def SystemSampling (totalPopulation sampleSize firstDrawn : ℕ) : 
  List ℕ := sorry

/-- Counts the number of elements in a list that fall within a given range -/
def CountInRange (list : List ℕ) (lower upper : ℕ) : ℕ := sorry

theorem questionnaire_C_count :
  let totalPopulation : ℕ := 960
  let sampleSize : ℕ := 32
  let firstDrawn : ℕ := 5
  let sample := SystemSampling totalPopulation sampleSize firstDrawn
  CountInRange sample 751 960 = 7 := by sorry

end questionnaire_C_count_l2785_278506


namespace five_topping_pizzas_l2785_278594

theorem five_topping_pizzas (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 5) :
  Nat.choose n k = 56 := by
  sorry

end five_topping_pizzas_l2785_278594


namespace smallest_k_for_p_cubed_minus_k_div_24_l2785_278557

-- Define p as the largest prime number with 1007 digits
def p : Nat := sorry

-- Define the property that p is prime
axiom p_is_prime : Nat.Prime p

-- Define the property that p has 1007 digits
axiom p_has_1007_digits : 10^1006 ≤ p ∧ p < 10^1007

-- Define the property that p is the largest such prime
axiom p_is_largest : ∀ q : Nat, Nat.Prime q → 10^1006 ≤ q ∧ q < 10^1007 → q ≤ p

-- Theorem statement
theorem smallest_k_for_p_cubed_minus_k_div_24 :
  (∃ k : Nat, k > 0 ∧ (p^3 - k) % 24 = 0) ∧
  (∀ k : Nat, k > 0 ∧ (p^3 - k) % 24 = 0 → k ≥ 1) :=
sorry

end smallest_k_for_p_cubed_minus_k_div_24_l2785_278557


namespace units_digit_of_quotient_units_digit_zero_l2785_278567

theorem units_digit_of_quotient (n : ℕ) : 
  (7^n + 4^n) % 9 = 2 :=
sorry

theorem units_digit_zero : 
  (7^2023 + 4^2023) / 9 % 10 = 0 :=
sorry

end units_digit_of_quotient_units_digit_zero_l2785_278567


namespace bucky_fish_count_l2785_278538

/-- The number of fish Bucky caught on Sunday -/
def F : ℕ := 5

/-- The price of the video game -/
def game_price : ℕ := 60

/-- The amount Bucky earned last weekend -/
def last_weekend_earnings : ℕ := 35

/-- The price of a trout -/
def trout_price : ℕ := 5

/-- The price of a blue-gill -/
def blue_gill_price : ℕ := 4

/-- The percentage of trout caught -/
def trout_percentage : ℚ := 3/5

/-- The percentage of blue-gill caught -/
def blue_gill_percentage : ℚ := 2/5

/-- The additional amount Bucky needs to save -/
def additional_savings : ℕ := 2

theorem bucky_fish_count :
  F * (trout_percentage * trout_price + blue_gill_percentage * blue_gill_price) =
  game_price - last_weekend_earnings - additional_savings :=
sorry

end bucky_fish_count_l2785_278538


namespace square_difference_equality_l2785_278558

theorem square_difference_equality : 1005^2 - 995^2 - 1002^2 + 996^2 = 8012 := by
  sorry

end square_difference_equality_l2785_278558


namespace least_positive_integer_multiple_l2785_278545

theorem least_positive_integer_multiple (x : ℕ) : x = 47 ↔ 
  (x > 0 ∧ ∀ y : ℕ, y > 0 → y < x → ¬((2*y)^2 + 2*47*2*y + 47^2) % 47 = 0) ∧
  ((2*x)^2 + 2*47*2*x + 47^2) % 47 = 0 :=
by sorry

end least_positive_integer_multiple_l2785_278545


namespace flower_purchase_cost_katie_flower_purchase_cost_l2785_278516

/-- The cost of buying roses and daisies at a fixed price per flower -/
theorem flower_purchase_cost 
  (price_per_flower : ℕ) 
  (num_roses : ℕ) 
  (num_daisies : ℕ) : 
  price_per_flower * (num_roses + num_daisies) = 
    price_per_flower * num_roses + price_per_flower * num_daisies :=
by sorry

/-- The total cost of Katie's flower purchase -/
theorem katie_flower_purchase_cost : 
  (5 : ℕ) + 5 = 10 ∧ 6 * 10 = 60 :=
by sorry

end flower_purchase_cost_katie_flower_purchase_cost_l2785_278516


namespace concert_duration_in_minutes_l2785_278583

/-- Converts hours and minutes to total minutes -/
def hours_minutes_to_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * 60 + minutes

/-- Theorem: A concert lasting 7 hours and 45 minutes is 465 minutes long -/
theorem concert_duration_in_minutes : 
  hours_minutes_to_minutes 7 45 = 465 := by
  sorry

end concert_duration_in_minutes_l2785_278583


namespace hyperbola_center_l2785_278549

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  center = (7, 2) ↔ f1 = (3, -2) ∧ f2 = (11, 6) := by
  sorry

#check hyperbola_center

end hyperbola_center_l2785_278549


namespace square_diagonal_point_theorem_l2785_278587

/-- Square with side length 12 -/
structure Square :=
  (side : ℝ)
  (is_twelve : side = 12)

/-- Point on the diagonal of the square -/
structure DiagonalPoint (s : Square) :=
  (x : ℝ)
  (y : ℝ)
  (on_diagonal : y = x)
  (in_square : 0 < x ∧ x < s.side)

/-- Circumcenter of a right triangle -/
def circumcenter (a b c : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem square_diagonal_point_theorem (s : Square) (p : DiagonalPoint s)
  (o1 : ℝ × ℝ) (o2 : ℝ × ℝ)
  (h1 : o1 = circumcenter (0, 0) (s.side, 0) (p.x, p.y))
  (h2 : o2 = circumcenter (s.side, s.side) (0, s.side) (p.x, p.y))
  (h3 : angle o1 (p.x, p.y) o2 = 120)
  : ∃ (a b : ℕ), (p.x : ℝ) = Real.sqrt a + Real.sqrt b ∧ a + b = 96 := by
  sorry

end square_diagonal_point_theorem_l2785_278587


namespace tim_cabinet_price_l2785_278532

/-- The price Tim paid for a cabinet after discount -/
theorem tim_cabinet_price (original_price : ℝ) (discount_percentage : ℝ) 
  (h1 : original_price = 1200)
  (h2 : discount_percentage = 15) : 
  original_price * (1 - discount_percentage / 100) = 1020 := by
  sorry

end tim_cabinet_price_l2785_278532


namespace largest_two_digit_prime_factor_of_binom_250_125_l2785_278584

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def is_two_digit_prime (p : ℕ) : Prop := 10 ≤ p ∧ p < 100 ∧ Nat.Prime p

theorem largest_two_digit_prime_factor_of_binom_250_125 :
  ∃ (p : ℕ), is_two_digit_prime p ∧
             p ∣ binomial_coefficient 250 125 ∧
             ∀ (q : ℕ), is_two_digit_prime q ∧ q ∣ binomial_coefficient 250 125 → q ≤ p ∧
             p = 83 :=
by sorry

end largest_two_digit_prime_factor_of_binom_250_125_l2785_278584


namespace x_range_l2785_278525

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 4| ≤ 6
def q (x : ℝ) : Prop := x^2 + 3*x ≥ 0

-- Define the theorem
theorem x_range :
  ∀ x : ℝ, (¬(p x ∧ q x) ∧ ¬(¬(p x))) → (-2 ≤ x ∧ x < 0) :=
by sorry

end x_range_l2785_278525


namespace inequality_implies_x_greater_than_one_l2785_278546

theorem inequality_implies_x_greater_than_one (x : ℝ) :
  x * (x^2 + 1) > (x + 1) * (x^2 - x + 1) → x > 1 := by
  sorry

end inequality_implies_x_greater_than_one_l2785_278546


namespace rectangle_properties_l2785_278524

/-- Properties of a rectangle with specific dimensions --/
theorem rectangle_properties (w : ℝ) (h : w > 0) :
  let l := 4 * w
  let perimeter := 2 * l + 2 * w
  perimeter = 200 →
  (l * w = 1600 ∧ perimeter - (perimeter - 5) = 5) := by
  sorry


end rectangle_properties_l2785_278524


namespace sum_lent_is_400_l2785_278535

/-- Prove that the sum lent is 400, given the conditions of the problem -/
theorem sum_lent_is_400 
  (interest_rate : ℚ) 
  (time_period : ℕ) 
  (interest_difference : ℚ) 
  (h1 : interest_rate = 4 / 100)
  (h2 : time_period = 8)
  (h3 : interest_difference = 272) :
  ∃ (sum_lent : ℚ), 
    sum_lent * interest_rate * time_period = sum_lent - interest_difference ∧ 
    sum_lent = 400 := by
  sorry

end sum_lent_is_400_l2785_278535


namespace single_digit_between_4_and_9_less_than_6_l2785_278592

theorem single_digit_between_4_and_9_less_than_6 (n : ℕ) 
  (h1 : n ≤ 9)
  (h2 : 4 < n)
  (h3 : n < 9)
  (h4 : n < 6) : 
  n = 5 := by
sorry

end single_digit_between_4_and_9_less_than_6_l2785_278592


namespace skincare_fraction_is_two_fifths_l2785_278564

/-- Represents Susie's babysitting and spending scenario -/
structure BabysittingScenario where
  hours_per_day : ℕ
  rate_per_hour : ℕ
  days_per_week : ℕ
  makeup_fraction : ℚ
  money_left : ℕ

/-- Calculates the fraction spent on skincare products given a babysitting scenario -/
def skincare_fraction (scenario : BabysittingScenario) : ℚ :=
  -- Definition to be proved
  2 / 5

/-- Theorem stating that given the specific scenario, the fraction spent on skincare is 2/5 -/
theorem skincare_fraction_is_two_fifths :
  let scenario : BabysittingScenario := {
    hours_per_day := 3,
    rate_per_hour := 10,
    days_per_week := 7,
    makeup_fraction := 3 / 10,
    money_left := 63
  }
  skincare_fraction scenario = 2 / 5 := by sorry

end skincare_fraction_is_two_fifths_l2785_278564


namespace factor_of_polynomial_l2785_278572

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (k : ℝ), 29 * 39 * x^4 + 4 = k * (x^2 - 2*x + 2) := by
  sorry

end factor_of_polynomial_l2785_278572


namespace correct_number_of_pitchers_l2785_278515

/-- The number of glasses each pitcher can serve -/
def glasses_per_pitcher : ℕ := 5

/-- The total number of glasses served -/
def total_glasses_served : ℕ := 30

/-- The number of pitchers prepared -/
def pitchers_prepared : ℕ := total_glasses_served / glasses_per_pitcher

theorem correct_number_of_pitchers : pitchers_prepared = 6 := by
  sorry

end correct_number_of_pitchers_l2785_278515


namespace gcf_of_180_150_210_l2785_278571

theorem gcf_of_180_150_210 : Nat.gcd 180 (Nat.gcd 150 210) = 30 := by
  sorry

end gcf_of_180_150_210_l2785_278571


namespace perfect_squares_identification_l2785_278595

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def option_A : ℕ := 3^4 * 4^5 * 7^7
def option_B : ℕ := 3^6 * 4^4 * 7^6
def option_C : ℕ := 3^5 * 4^6 * 7^5
def option_D : ℕ := 3^4 * 4^7 * 7^4
def option_E : ℕ := 3^6 * 4^6 * 7^6

theorem perfect_squares_identification :
  ¬(is_perfect_square option_A) ∧
  (is_perfect_square option_B) ∧
  ¬(is_perfect_square option_C) ∧
  (is_perfect_square option_D) ∧
  (is_perfect_square option_E) :=
by sorry

end perfect_squares_identification_l2785_278595


namespace inequality1_solution_inequality2_solution_l2785_278530

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := -x^2 + 4*x + 5 < 0
def inequality2 (x : ℝ) : Prop := 2*x^2 - 5*x + 2 ≤ 0

-- Define the solution sets
def solution_set1 : Set ℝ := {x | x < -1 ∨ x > 5}
def solution_set2 : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}

-- Theorem statements
theorem inequality1_solution : 
  ∀ x : ℝ, inequality1 x ↔ x ∈ solution_set1 := by sorry

theorem inequality2_solution : 
  ∀ x : ℝ, inequality2 x ↔ x ∈ solution_set2 := by sorry

end inequality1_solution_inequality2_solution_l2785_278530


namespace minimum_value_a_l2785_278542

theorem minimum_value_a (a : ℝ) : (∀ x₁ x₂ x₃ x₄ : ℝ, ∃ k₁ k₂ k₃ k₄ : ℤ,
  (x₁ - k₁ - (x₂ - k₂))^2 + (x₁ - k₁ - (x₃ - k₃))^2 + (x₁ - k₁ - (x₄ - k₄))^2 +
  (x₂ - k₂ - (x₃ - k₃))^2 + (x₂ - k₂ - (x₄ - k₄))^2 + (x₃ - k₃ - (x₄ - k₄))^2 ≤ a) →
  a ≥ 5/4 :=
by sorry

end minimum_value_a_l2785_278542


namespace rectangle_area_l2785_278502

/-- The area of a rectangle with length 15 cm and width 0.9 times its length is 202.5 cm². -/
theorem rectangle_area : 
  let length : ℝ := 15
  let width : ℝ := 0.9 * length
  length * width = 202.5 := by
sorry

end rectangle_area_l2785_278502


namespace divisibility_by_three_l2785_278555

theorem divisibility_by_three (x y : ℤ) : 
  (3 ∣ x^2 + y^2) → (3 ∣ x) ∧ (3 ∣ y) := by
  sorry

end divisibility_by_three_l2785_278555


namespace x_cube_minus_3x_eq_6_l2785_278537

theorem x_cube_minus_3x_eq_6 (x : ℝ) (h : x^3 - 3*x = 6) :
  x^6 + 27*x^2 = 36*x^2 + 36*x + 36 := by
  sorry

end x_cube_minus_3x_eq_6_l2785_278537


namespace tammy_caught_30_times_l2785_278529

/-- Calculates the number of times Tammy caught the ball given the conditions of the problem. -/
def tammys_catches (joe_catches : ℕ) : ℕ :=
  let derek_catches := 2 * joe_catches - 4
  let tammy_catches := derek_catches / 3 + 16
  tammy_catches

/-- Theorem stating that Tammy caught the ball 30 times given the problem conditions. -/
theorem tammy_caught_30_times : tammys_catches 23 = 30 := by
  sorry

#eval tammys_catches 23

end tammy_caught_30_times_l2785_278529


namespace gmat_question_percentage_l2785_278569

/-- The percentage of test takers who answered the second question correctly -/
def second_correct : ℝ := 75

/-- The percentage of test takers who answered neither question correctly -/
def neither_correct : ℝ := 5

/-- The percentage of test takers who answered both questions correctly -/
def both_correct : ℝ := 60

/-- The percentage of test takers who answered the first question correctly -/
def first_correct : ℝ := 80

theorem gmat_question_percentage :
  first_correct = 80 :=
sorry

end gmat_question_percentage_l2785_278569


namespace set_operations_l2785_278562

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

def B : Set ℝ := {x : ℝ | x < -3 ∨ x > 1}

theorem set_operations :
  (A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2}) ∧
  (A ∪ B = {x : ℝ | x < -3 ∨ x > 0}) ∧
  ((Aᶜ) ∩ (Bᶜ) = {x : ℝ | -3 ≤ x ∧ x ≤ 0}) := by sorry

end set_operations_l2785_278562


namespace min_value_of_s_min_value_of_s_achieved_l2785_278536

theorem min_value_of_s (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (y*z/x + z*x/y + x*y/z) ≥ Real.sqrt 3 := by
  sorry

theorem min_value_of_s_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 1 ∧
    b*c/a + c*a/b + a*b/c = Real.sqrt 3 := by
  sorry

end min_value_of_s_min_value_of_s_achieved_l2785_278536


namespace root_difference_quadratic_equation_l2785_278503

theorem root_difference_quadratic_equation : ∃ (x y : ℝ), 
  (x^2 + 40*x + 300 = -48) ∧ 
  (y^2 + 40*y + 300 = -48) ∧ 
  x ≠ y ∧ 
  |x - y| = 16 := by
sorry

end root_difference_quadratic_equation_l2785_278503


namespace valid_marking_exists_l2785_278551

/-- Represents a marking of cells in a 9x9 table -/
def Marking := Fin 9 → Fin 9 → Bool

/-- Checks if two adjacent rows have at least 6 marked cells -/
def validRows (m : Marking) : Prop :=
  ∀ i : Fin 8, (Finset.sum (Finset.univ.filter (λ j => m i j || m (i + 1) j)) (λ _ => 1) : ℕ) ≥ 6

/-- Checks if two adjacent columns have at most 5 marked cells -/
def validColumns (m : Marking) : Prop :=
  ∀ j : Fin 8, (Finset.sum (Finset.univ.filter (λ i => m i j || m i (j + 1))) (λ _ => 1) : ℕ) ≤ 5

/-- Theorem stating that a valid marking exists -/
theorem valid_marking_exists : ∃ m : Marking, validRows m ∧ validColumns m := by
  sorry

end valid_marking_exists_l2785_278551


namespace fraction_of_total_l2785_278540

theorem fraction_of_total (total : ℝ) (r_amount : ℝ) (h1 : total = 5000) (h2 : r_amount = 2000.0000000000002) :
  r_amount / total = 0.40000000000000004 := by
  sorry

end fraction_of_total_l2785_278540


namespace square_triangle_equal_area_l2785_278550

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) : 
  square_perimeter = 48 →
  triangle_height = 48 →
  (square_perimeter / 4)^2 = (1/2) * x * triangle_height →
  x = 6 := by
  sorry

end square_triangle_equal_area_l2785_278550


namespace fraction_addition_l2785_278553

theorem fraction_addition : (4 / 7 : ℚ) / 5 + 1 / 3 = 47 / 105 := by sorry

end fraction_addition_l2785_278553


namespace cookie_problem_l2785_278510

theorem cookie_problem (C : ℕ) : C ≥ 187 ∧ (3 : ℚ) / 70 * C = 8 → C = 187 :=
by
  sorry

#check cookie_problem

end cookie_problem_l2785_278510


namespace stone_density_l2785_278522

/-- Given a cylindrical container with water and a stone, this theorem relates
    the density of the stone to the water level changes under different conditions. -/
theorem stone_density (S : ℝ) (ρ h₁ h₂ : ℝ) (hS : S > 0) (hρ : ρ > 0) (hh₁ : h₁ > 0) (hh₂ : h₂ > 0) :
  let ρ_s := (ρ * h₁) / h₂
  ρ_s = (ρ * S * h₁) / (S * h₂) :=
by sorry

end stone_density_l2785_278522


namespace hyperbola_eccentricity_l2785_278565

/-- Represents a hyperbola with center O and focus F -/
structure Hyperbola where
  O : ℝ × ℝ  -- Center of the hyperbola
  F : ℝ × ℝ  -- Focus of the hyperbola

/-- Represents a point on the asymptote of the hyperbola -/
def AsymptoticPoint (h : Hyperbola) := ℝ × ℝ

/-- Checks if a triangle is isosceles right -/
def IsIsoscelesRight (A B C : ℝ × ℝ) : Prop := sorry

/-- Calculates the eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Main theorem: If a point P on the asymptote of a hyperbola forms an isosceles right triangle
    with the center O and focus F, then the eccentricity of the hyperbola is √2 -/
theorem hyperbola_eccentricity 
  (h : Hyperbola) 
  (P : AsymptoticPoint h) 
  (h_isosceles : IsIsoscelesRight h.O h.F P) : 
  eccentricity h = Real.sqrt 2 := by sorry

end hyperbola_eccentricity_l2785_278565


namespace abc_equality_l2785_278586

theorem abc_equality (a b c : ℕ) 
  (h : ∀ n : ℕ, (a * b * c)^n ∣ ((a^n - 1) * (b^n - 1) * (c^n - 1) + 1)^3) : 
  a = b ∧ b = c :=
sorry

end abc_equality_l2785_278586


namespace point_inside_circle_parameter_range_l2785_278528

theorem point_inside_circle_parameter_range :
  ∀ a : ℝ, 
  (∃ x y : ℝ, (x - a)^2 + (y + a)^2 = 4 ∧ (1 - a)^2 + (1 + a)^2 < 4) →
  -1 < a ∧ a < 1 :=
by sorry

end point_inside_circle_parameter_range_l2785_278528


namespace pencils_per_pack_l2785_278589

theorem pencils_per_pack (num_packs : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) : 
  num_packs = 35 → num_rows = 70 → pencils_per_row = 2 → 
  (num_rows * pencils_per_row) / num_packs = 4 := by
sorry

end pencils_per_pack_l2785_278589


namespace quadratic_equation_roots_l2785_278539

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 - 2*(m-2)*x₁ + m^2 = 0 ∧ 
   x₂^2 - 2*(m-2)*x₂ + m^2 = 0) → 
  m < 1 := by
sorry

end quadratic_equation_roots_l2785_278539


namespace current_books_count_l2785_278509

/-- The number of books in a library over time -/
def library_books (initial_old_books : ℕ) (bought_two_years_ago : ℕ) (bought_last_year : ℕ) (donated_this_year : ℕ) : ℕ :=
  initial_old_books + bought_two_years_ago + bought_last_year - donated_this_year

/-- Theorem: The current number of books in the library is 1000 -/
theorem current_books_count :
  let initial_old_books : ℕ := 500
  let bought_two_years_ago : ℕ := 300
  let bought_last_year : ℕ := bought_two_years_ago + 100
  let donated_this_year : ℕ := 200
  library_books initial_old_books bought_two_years_ago bought_last_year donated_this_year = 1000 := by
  sorry

end current_books_count_l2785_278509


namespace chord_length_polar_l2785_278576

/-- Chord length intercepted by a line on a circle in polar coordinates -/
theorem chord_length_polar (ρ θ : ℝ) (h1 : ρ = 4 * Real.sin θ) (h2 : Real.tan θ = 1/2) :
  ρ = 4 * Real.sqrt 5 / 5 := by
  sorry

end chord_length_polar_l2785_278576


namespace coinciding_rest_days_theorem_l2785_278599

/-- Charlie's schedule cycle length -/
def charlie_cycle : Nat := 6

/-- Dana's schedule cycle length -/
def dana_cycle : Nat := 10

/-- Number of days in the period -/
def total_days : Nat := 1200

/-- Number of rest days in Charlie's cycle -/
def charlie_rest_days : Nat := 2

/-- Number of rest days in Dana's cycle -/
def dana_rest_days : Nat := 1

/-- Function to calculate the number of coinciding rest days -/
def coinciding_rest_days (charlie_cycle dana_cycle total_days : Nat) : Nat :=
  sorry

theorem coinciding_rest_days_theorem :
  coinciding_rest_days charlie_cycle dana_cycle total_days = 40 := by
  sorry

end coinciding_rest_days_theorem_l2785_278599


namespace chess_tournament_games_l2785_278561

/-- The number of games played in a chess tournament with n participants,
    where each participant plays exactly one game with each other participant. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 22 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 231. -/
theorem chess_tournament_games :
  tournament_games 22 = 231 := by sorry

end chess_tournament_games_l2785_278561


namespace students_playing_both_sports_l2785_278552

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 40 →
  football = 26 →
  tennis = 20 →
  neither = 11 →
  ∃ both : ℕ, both = 17 ∧ total = football + tennis - both + neither :=
by sorry

end students_playing_both_sports_l2785_278552


namespace problem_statement_l2785_278513

theorem problem_statement : (481 + 426)^2 - 4 * 481 * 426 = 3025 := by
  sorry

end problem_statement_l2785_278513


namespace hotel_reunion_attendees_l2785_278527

theorem hotel_reunion_attendees (total_guests oates_attendees hall_attendees : ℕ) 
  (h1 : total_guests = 100)
  (h2 : oates_attendees = 40)
  (h3 : hall_attendees = 70)
  (h4 : total_guests ≤ oates_attendees + hall_attendees) :
  oates_attendees + hall_attendees - total_guests = 10 := by
  sorry

end hotel_reunion_attendees_l2785_278527


namespace kitten_price_l2785_278574

theorem kitten_price (kitten_count puppy_count : ℕ) 
                     (puppy_price total_earnings : ℚ) :
  kitten_count = 2 →
  puppy_count = 1 →
  puppy_price = 5 →
  total_earnings = 17 →
  ∃ kitten_price : ℚ, 
    kitten_price * kitten_count + puppy_price * puppy_count = total_earnings ∧
    kitten_price = 6 :=
by sorry

end kitten_price_l2785_278574


namespace worker_count_l2785_278559

theorem worker_count (total : ℕ) (increased_total : ℕ) (extra_contribution : ℕ) : 
  (total = 300000) → 
  (increased_total = 325000) → 
  (extra_contribution = 50) → 
  (∃ (n : ℕ), (n * (total / n) = total) ∧ 
              (n * (total / n + extra_contribution) = increased_total) ∧ 
              (n = 500)) := by
  sorry

end worker_count_l2785_278559


namespace sons_age_l2785_278541

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 34 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 32 := by
sorry

end sons_age_l2785_278541


namespace largest_increase_1998_l2785_278581

def sales : ℕ → ℕ
| 0 => 3000    -- 1994
| 1 => 4500    -- 1995
| 2 => 6000    -- 1996
| 3 => 6750    -- 1997
| 4 => 8400    -- 1998
| 5 => 9000    -- 1999
| 6 => 9600    -- 2000
| 7 => 10400   -- 2001
| 8 => 9500    -- 2002
| 9 => 6500    -- 2003
| _ => 0       -- undefined for other years

def salesIncrease (year : ℕ) : ℤ :=
  (sales (year + 1) : ℤ) - (sales year : ℤ)

theorem largest_increase_1998 :
  ∀ y : ℕ, y ≥ 0 ∧ y < 9 → salesIncrease 4 ≥ salesIncrease y :=
by sorry

end largest_increase_1998_l2785_278581


namespace math_team_selection_l2785_278533

theorem math_team_selection (boys : ℕ) (girls : ℕ) (team_size : ℕ) : 
  boys = 10 → girls = 12 → team_size = 8 → 
  Nat.choose (boys + girls) team_size = 319770 := by
sorry

end math_team_selection_l2785_278533


namespace tan_alpha_minus_pi_fourth_l2785_278508

theorem tan_alpha_minus_pi_fourth (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α = 3/5) : 
  Real.tan (α - Real.pi/4) = -1/7 ∨ Real.tan (α - Real.pi/4) = -7 := by
  sorry

end tan_alpha_minus_pi_fourth_l2785_278508


namespace interview_score_calculation_l2785_278517

/-- Calculate the interview score based on individual scores and their proportions -/
theorem interview_score_calculation 
  (basic_knowledge : ℝ) 
  (communication_skills : ℝ) 
  (work_attitude : ℝ) 
  (basic_knowledge_proportion : ℝ) 
  (communication_skills_proportion : ℝ) 
  (work_attitude_proportion : ℝ) 
  (h1 : basic_knowledge = 92) 
  (h2 : communication_skills = 87) 
  (h3 : work_attitude = 94) 
  (h4 : basic_knowledge_proportion = 0.2) 
  (h5 : communication_skills_proportion = 0.3) 
  (h6 : work_attitude_proportion = 0.5) :
  basic_knowledge * basic_knowledge_proportion + 
  communication_skills * communication_skills_proportion + 
  work_attitude * work_attitude_proportion = 91.5 := by
sorry

end interview_score_calculation_l2785_278517


namespace angle_terminal_side_value_l2785_278514

theorem angle_terminal_side_value (a : ℝ) (h : a > 0) :
  let x := 5 * a
  let y := -12 * a
  let r := Real.sqrt (x^2 + y^2)
  let sinα := y / r
  let cosα := x / r
  2 * sinα + cosα = -19 / 13 := by sorry

end angle_terminal_side_value_l2785_278514


namespace inequality_preservation_l2785_278544

theorem inequality_preservation (a b c : ℝ) (h : a > b) : c - a < c - b := by
  sorry

end inequality_preservation_l2785_278544


namespace imaginary_part_of_complex_number_l2785_278577

theorem imaginary_part_of_complex_number (z : ℂ) : z = (3 - 2 * Complex.I^2) / (1 + Complex.I) → z.im = -5/2 := by
  sorry

end imaginary_part_of_complex_number_l2785_278577


namespace necessary_but_not_sufficient_condition_l2785_278505

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a > b → a - b > -2) ∧ ¬(a - b > -2 → a > b) :=
sorry

end necessary_but_not_sufficient_condition_l2785_278505


namespace apple_cost_price_l2785_278568

-- Define the selling price
def selling_price : ℚ := 19

-- Define the ratio of selling price to cost price
def selling_to_cost_ratio : ℚ := 5/6

-- Theorem statement
theorem apple_cost_price :
  ∃ (cost_price : ℚ), 
    cost_price = selling_price / selling_to_cost_ratio ∧ 
    cost_price = 114/5 := by
  sorry

end apple_cost_price_l2785_278568


namespace r_l2785_278570

/-- r'(n) is the sum of distinct primes in the prime factorization of n -/
noncomputable def r' (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

/-- The set of composite positive integers -/
def CompositeSet : Set ℕ :=
  {n : ℕ | n > 1 ∧ ¬Nat.Prime n}

/-- The set of integers that can be expressed as sums of two or more distinct primes -/
def SumOfDistinctPrimesSet : Set ℕ :=
  {n : ℕ | ∃ (s : Finset ℕ), s.card ≥ 2 ∧ (∀ p ∈ s, Nat.Prime p) ∧ s.sum id = n}

/-- The range of r' is equal to the set of integers that can be expressed as sums of two or more distinct primes -/
theorem r'_range_eq_sum_of_distinct_primes :
  (CompositeSet.image r') = SumOfDistinctPrimesSet :=
sorry

end r_l2785_278570


namespace problem_statement_l2785_278556

theorem problem_statement : 2 * ((40 / 8) + (34 / 12)) = 14 := by
  sorry

end problem_statement_l2785_278556


namespace soda_consumption_per_person_l2785_278519

/- Define the problem parameters -/
def people_attending : ℕ := 5 * 12  -- five dozens
def cans_per_box : ℕ := 10
def cost_per_box : ℕ := 2
def family_members : ℕ := 6
def payment_per_member : ℕ := 4

/- Define the theorem -/
theorem soda_consumption_per_person :
  let total_payment := family_members * payment_per_member
  let boxes_bought := total_payment / cost_per_box
  let total_cans := boxes_bought * cans_per_box
  total_cans / people_attending = 2 := by sorry

end soda_consumption_per_person_l2785_278519


namespace train_speed_l2785_278563

/-- Proves that a train with given length, crossing a platform of given length in a given time, has a specific speed in km/h -/
theorem train_speed (train_length platform_length : Real) (crossing_time : Real) : 
  train_length = 450 ∧ 
  platform_length = 250.056 ∧ 
  crossing_time = 20 →
  (train_length + platform_length) / crossing_time * 3.6 = 126.01008 := by
  sorry


end train_speed_l2785_278563


namespace black_balls_probability_l2785_278504

theorem black_balls_probability 
  (m₁ m₂ k₁ k₂ : ℕ) 
  (h_total : m₁ + m₂ = 25)
  (h_white_prob : (k₁ : ℝ) / m₁ * (k₂ : ℝ) / m₂ = 0.54)
  : ((m₁ - k₁ : ℝ) / m₁) * ((m₂ - k₂ : ℝ) / m₂) = 0.04 := by
  sorry

end black_balls_probability_l2785_278504


namespace expansion_terms_count_l2785_278582

/-- The number of terms in the expansion of a product of two polynomials with distinct variables -/
def num_terms_in_expansion (n m : ℕ) : ℕ := n * m

theorem expansion_terms_count : 
  let first_factor_terms : ℕ := 3
  let second_factor_terms : ℕ := 6
  num_terms_in_expansion first_factor_terms second_factor_terms = 18 := by sorry

end expansion_terms_count_l2785_278582


namespace isabel_sold_three_bead_necklaces_total_cost_equals_earnings_l2785_278588

/-- The number of bead necklaces sold by Isabel -/
def bead_necklaces : ℕ := sorry

/-- The number of gem stone necklaces sold by Isabel -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 6

/-- The total earnings from all necklaces in dollars -/
def total_earnings : ℕ := 36

/-- Theorem stating that Isabel sold 3 bead necklaces -/
theorem isabel_sold_three_bead_necklaces :
  bead_necklaces = 3 :=
by
  sorry

/-- The total number of necklaces sold -/
def total_necklaces : ℕ := bead_necklaces + gem_necklaces

/-- The total cost of all necklaces sold -/
def total_cost : ℕ := total_necklaces * necklace_cost

/-- Assertion that the total cost equals the total earnings -/
theorem total_cost_equals_earnings :
  total_cost = total_earnings :=
by
  sorry

end isabel_sold_three_bead_necklaces_total_cost_equals_earnings_l2785_278588


namespace absolute_prime_at_most_three_digits_l2785_278585

/-- A function that returns true if a positive integer is prime -/
def IsPrime (n : ℕ) : Prop := sorry

/-- A function that returns the set of distinct digits in a positive integer's decimal representation -/
def DistinctDigits (n : ℕ) : Finset ℕ := sorry

/-- A function that returns true if all permutations of a positive integer's digits are prime -/
def AllDigitPermutationsPrime (n : ℕ) : Prop := sorry

/-- Definition of an absolute prime -/
def IsAbsolutePrime (n : ℕ) : Prop :=
  n > 0 ∧ IsPrime n ∧ AllDigitPermutationsPrime n

theorem absolute_prime_at_most_three_digits (n : ℕ) :
  IsAbsolutePrime n → Finset.card (DistinctDigits n) ≤ 3 := by sorry

end absolute_prime_at_most_three_digits_l2785_278585


namespace marble_count_l2785_278597

/-- Given a bag of marbles with blue, red, and white marbles, 
    prove that the total number of marbles is 50 -/
theorem marble_count (blue red white : ℕ) (total : ℕ) 
    (h1 : blue = 5)
    (h2 : red = 9)
    (h3 : total = blue + red + white)
    (h4 : (red + white : ℚ) / total = 9/10) :
  total = 50 := by
  sorry

end marble_count_l2785_278597


namespace equation_solution_l2785_278543

theorem equation_solution (x : ℝ) (h : x > 0) :
  (x - 3) / 8 = 5 / (x - 8) ↔ x = 16 := by sorry

end equation_solution_l2785_278543
