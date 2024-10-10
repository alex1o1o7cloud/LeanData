import Mathlib

namespace arithmetic_sequence_count_l1228_122876

theorem arithmetic_sequence_count : 
  let a₁ : ℝ := 2.5
  let d : ℝ := 5
  let aₙ : ℝ := 62.5
  (aₙ - a₁) / d + 1 = 13 := by
  sorry

end arithmetic_sequence_count_l1228_122876


namespace polar_to_rectangular_equivalence_l1228_122894

-- Define the polar coordinate equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Define the rectangular coordinate equation
def rectangular_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Theorem stating the equivalence of the two equations
theorem polar_to_rectangular_equivalence :
  ∀ (x y ρ θ : ℝ), 
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (polar_equation ρ θ ↔ rectangular_equation x y) :=
sorry

end polar_to_rectangular_equivalence_l1228_122894


namespace sin_210_degrees_l1228_122830

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end sin_210_degrees_l1228_122830


namespace not_parabola_l1228_122855

/-- A conic section represented by the equation x^2 + ky^2 = 1 -/
structure ConicSection (k : ℝ) where
  x : ℝ
  y : ℝ
  eq : x^2 + k * y^2 = 1

/-- Definition of a parabola -/
def IsParabola (c : ConicSection k) : Prop :=
  ∃ (a b h : ℝ), h ≠ 0 ∧ (c.x - a)^2 = 4 * h * (c.y - b)

/-- Theorem: For any real k, the equation x^2 + ky^2 = 1 cannot represent a parabola -/
theorem not_parabola (k : ℝ) : ¬∃ (c : ConicSection k), IsParabola c := by
  sorry

end not_parabola_l1228_122855


namespace min_packs_for_120_cans_l1228_122827

/-- Represents a combination of soda packs -/
structure SodaPacks where
  pack8 : ℕ
  pack15 : ℕ
  pack32 : ℕ

/-- Calculates the total number of cans for a given combination of packs -/
def totalCans (packs : SodaPacks) : ℕ :=
  8 * packs.pack8 + 15 * packs.pack15 + 32 * packs.pack32

/-- Calculates the total number of packs for a given combination -/
def totalPacks (packs : SodaPacks) : ℕ :=
  packs.pack8 + packs.pack15 + packs.pack32

/-- Theorem: The minimum number of packs to buy exactly 120 cans is 6 -/
theorem min_packs_for_120_cans : 
  ∃ (min_packs : SodaPacks), 
    totalCans min_packs = 120 ∧ 
    totalPacks min_packs = 6 ∧
    ∀ (other_packs : SodaPacks), 
      totalCans other_packs = 120 → 
      totalPacks other_packs ≥ totalPacks min_packs :=
by
  sorry

end min_packs_for_120_cans_l1228_122827


namespace total_results_l1228_122801

theorem total_results (average : ℝ) (first_five_avg : ℝ) (last_seven_avg : ℝ) (fifth_result : ℝ)
  (h1 : average = 42)
  (h2 : first_five_avg = 49)
  (h3 : last_seven_avg = 52)
  (h4 : fifth_result = 147) :
  ∃ n : ℕ, n = 11 ∧ n * average = 5 * first_five_avg + 7 * last_seven_avg - fifth_result := by
  sorry

end total_results_l1228_122801


namespace jacks_second_half_time_l1228_122804

/-- Proves that Jack's time for the second half of the hill is 6 seconds -/
theorem jacks_second_half_time
  (jack_first_half : ℕ)
  (jack_finishes_before : ℕ)
  (jill_total_time : ℕ)
  (h1 : jack_first_half = 19)
  (h2 : jack_finishes_before = 7)
  (h3 : jill_total_time = 32) :
  jill_total_time - jack_finishes_before - jack_first_half = 6 := by
  sorry

#check jacks_second_half_time

end jacks_second_half_time_l1228_122804


namespace rectangle_ratio_l1228_122896

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0)
  (h4 : s + 2*y = 3*s) -- Outer square side length
  (h5 : x + y = 3*s) -- Outer square side length
  (h6 : (3*s)^2 = 9*s^2) -- Area of outer square is 9 times inner square
  : x / y = 2 := by
sorry

end rectangle_ratio_l1228_122896


namespace school_bus_seats_l1228_122862

/-- Proves that the number of seats on each school bus is 9, given the conditions of the field trip. -/
theorem school_bus_seats (total_students : ℕ) (num_buses : ℕ) (h1 : total_students = 45) (h2 : num_buses = 5) (h3 : total_students % num_buses = 0) :
  total_students / num_buses = 9 := by
sorry

end school_bus_seats_l1228_122862


namespace outfit_choices_l1228_122892

theorem outfit_choices (shirts : ℕ) (skirts : ℕ) (dresses : ℕ) : 
  shirts = 4 → skirts = 3 → dresses = 2 → shirts * skirts + dresses = 14 := by
  sorry

end outfit_choices_l1228_122892


namespace elevator_max_weight_next_person_l1228_122879

/-- Given an elevator scenario with adults and children, calculate the maximum weight of the next person that can enter without overloading the elevator. -/
theorem elevator_max_weight_next_person 
  (num_adults : ℕ) 
  (avg_weight_adults : ℝ) 
  (num_children : ℕ) 
  (avg_weight_children : ℝ) 
  (max_elevator_weight : ℝ) 
  (h1 : num_adults = 7) 
  (h2 : avg_weight_adults = 150) 
  (h3 : num_children = 5) 
  (h4 : avg_weight_children = 70) 
  (h5 : max_elevator_weight = 1500) :
  max_elevator_weight - (num_adults * avg_weight_adults + num_children * avg_weight_children) = 100 := by
  sorry

end elevator_max_weight_next_person_l1228_122879


namespace no_nonzero_ending_product_zero_l1228_122887

theorem no_nonzero_ending_product_zero (x y : ℤ) : 
  (x % 10 ≠ 0) → (y % 10 ≠ 0) → (x * y ≠ 0) :=
by sorry

end no_nonzero_ending_product_zero_l1228_122887


namespace shopping_theorem_l1228_122852

def shopping_calculation (initial_amount : ℝ) 
  (baguette_cost : ℝ) (baguette_quantity : ℕ)
  (water_cost : ℝ) (water_quantity : ℕ)
  (chocolate_cost : ℝ) (chocolate_quantity : ℕ)
  (milk_cost : ℝ) (milk_discount : ℝ)
  (chips_cost : ℝ) (chips_discount : ℝ)
  (sales_tax : ℝ) : ℝ :=
  let baguette_total := baguette_cost * baguette_quantity
  let water_total := water_cost * water_quantity
  let chocolate_total := (chocolate_cost * 2) * 0.8 * (1 + sales_tax)
  let milk_total := milk_cost * (1 - milk_discount)
  let chips_total := (chips_cost + chips_cost * chips_discount) * (1 + sales_tax)
  let total_cost := baguette_total + water_total + chocolate_total + milk_total + chips_total
  initial_amount - total_cost

theorem shopping_theorem : 
  shopping_calculation 50 2 2 1 2 1.5 3 3.5 0.1 2.5 0.5 0.08 = 34.208 := by
  sorry

end shopping_theorem_l1228_122852


namespace payment_ways_formula_l1228_122839

/-- The number of ways to pay n euros using 1-euro and 2-euro coins -/
def paymentWays (n : ℕ) : ℕ := n / 2 + 1

/-- Theorem: The number of ways to pay n euros using 1-euro and 2-euro coins
    is equal to ⌊n/2⌋ + 1 -/
theorem payment_ways_formula (n : ℕ) :
  paymentWays n = n / 2 + 1 := by
  sorry

#check payment_ways_formula

end payment_ways_formula_l1228_122839


namespace triangle_equality_l1228_122873

/-- Given a triangle ABC with sides a, b, c opposite to angles α, β, γ respectively,
    and circumradius R, prove that if the given equation holds, then the triangle is equilateral. -/
theorem triangle_equality (a b c R : ℝ) (α β γ : ℝ) : 
  a > 0 → b > 0 → c > 0 → R > 0 →
  0 < α ∧ α < π → 0 < β ∧ β < π → 0 < γ ∧ γ < π →
  α + β + γ = π →
  (a * Real.cos α + b * Real.cos β + c * Real.cos γ) / (a * Real.sin β + b * Real.sin γ + c * Real.sin α) = (a + b + c) / (9 * R) →
  α = β ∧ β = γ ∧ γ = π / 3 := by
  sorry

end triangle_equality_l1228_122873


namespace count_white_rhinos_l1228_122859

/-- Given information about rhinos and their weights, prove the number of white rhinos --/
theorem count_white_rhinos (white_rhino_weight : ℕ) (black_rhino_count : ℕ) (black_rhino_weight : ℕ) (total_weight : ℕ) : 
  white_rhino_weight = 5100 →
  black_rhino_count = 8 →
  black_rhino_weight = 2000 →
  total_weight = 51700 →
  (total_weight - black_rhino_count * black_rhino_weight) / white_rhino_weight = 7 := by
sorry

end count_white_rhinos_l1228_122859


namespace solve_cubic_equation_l1228_122895

theorem solve_cubic_equation :
  ∃ y : ℝ, (y - 5)^3 = (1/27)⁻¹ ∧ y = 8 := by sorry

end solve_cubic_equation_l1228_122895


namespace max_value_inequality_l1228_122881

theorem max_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ k : ℝ, (a + b + c) * (1 / a + 1 / (b + c)) ≥ k) ↔ k ≤ 4 :=
sorry

end max_value_inequality_l1228_122881


namespace gcd_45345_34534_l1228_122868

theorem gcd_45345_34534 : Nat.gcd 45345 34534 = 71 := by
  sorry

end gcd_45345_34534_l1228_122868


namespace charles_total_earnings_l1228_122882

/-- Calculates Charles's total earnings from housesitting and dog walking -/
def charles_earnings (housesit_rate : ℕ) (dog_walk_rate : ℕ) (housesit_hours : ℕ) (dogs_walked : ℕ) : ℕ :=
  housesit_rate * housesit_hours + dog_walk_rate * dogs_walked

/-- Theorem stating that Charles's earnings are $216 given the specified rates and hours -/
theorem charles_total_earnings :
  charles_earnings 15 22 10 3 = 216 := by
  sorry

end charles_total_earnings_l1228_122882


namespace swimming_speed_calculation_l1228_122899

/-- Represents the swimming scenario with a stream -/
structure SwimmingScenario where
  stream_speed : ℝ
  upstream_time : ℝ
  downstream_time : ℝ
  swimming_speed : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : SwimmingScenario) : Prop :=
  s.stream_speed = 3 ∧ s.upstream_time = 2 * s.downstream_time

/-- The theorem to be proved -/
theorem swimming_speed_calculation (s : SwimmingScenario) :
  problem_conditions s → s.swimming_speed = 9 := by
  sorry


end swimming_speed_calculation_l1228_122899


namespace power_eight_mod_five_l1228_122872

theorem power_eight_mod_five : 8^2023 % 5 = 2 := by
  sorry

end power_eight_mod_five_l1228_122872


namespace line_slope_proof_l1228_122854

/-- Given two points (a, -1) and (2, 3) on a line with slope 2, prove that a = 0 -/
theorem line_slope_proof (a : ℝ) : 
  (3 - (-1)) / (2 - a) = 2 → a = 0 := by
  sorry

end line_slope_proof_l1228_122854


namespace total_food_count_l1228_122820

/-- The total number of hotdogs and hamburgers brought by neighbors -/
theorem total_food_count : ℕ := by
  -- Define the number of hotdogs brought by each neighbor
  let first_neighbor_hotdogs : ℕ := 75
  let second_neighbor_hotdogs : ℕ := first_neighbor_hotdogs - 25
  let third_neighbor_hotdogs : ℕ := 35
  let fourth_neighbor_hotdogs : ℕ := 2 * third_neighbor_hotdogs

  -- Define the number of hamburgers brought
  let one_neighbor_hamburgers : ℕ := 60
  let another_neighbor_hamburgers : ℕ := 3 * one_neighbor_hamburgers

  -- Calculate total hotdogs and hamburgers
  let total_hotdogs : ℕ := first_neighbor_hotdogs + second_neighbor_hotdogs + 
                           third_neighbor_hotdogs + fourth_neighbor_hotdogs
  let total_hamburgers : ℕ := one_neighbor_hamburgers + another_neighbor_hamburgers
  let total_food : ℕ := total_hotdogs + total_hamburgers

  -- Prove that the total is 470
  have : total_food = 470 := by sorry

  exact 470

end total_food_count_l1228_122820


namespace sphere_centers_distance_l1228_122874

/-- The distance between the centers of two spheres with masses M and m, 
    where a point B exists such that both spheres exert equal gravitational force on it,
    and A is a point between the centers with distance d from B. -/
theorem sphere_centers_distance (M m d : ℝ) (hM : M > 0) (hm : m > 0) (hd : d > 0) : 
  ∃ (distance : ℝ), distance = d / 2 * (M - m) / Real.sqrt (M * m) :=
sorry

end sphere_centers_distance_l1228_122874


namespace mary_chopped_chairs_l1228_122823

/-- Represents the number of sticks of wood produced by different furniture types -/
structure FurnitureWood where
  chair : ℕ
  table : ℕ
  stool : ℕ

/-- Represents the furniture Mary chopped up -/
structure ChoppedFurniture where
  chairs : ℕ
  tables : ℕ
  stools : ℕ

/-- Calculates the total number of sticks from chopped furniture -/
def totalSticks (fw : FurnitureWood) (cf : ChoppedFurniture) : ℕ :=
  fw.chair * cf.chairs + fw.table * cf.tables + fw.stool * cf.stools

theorem mary_chopped_chairs :
  ∀ (fw : FurnitureWood) (cf : ChoppedFurniture) (burn_rate hours_warm : ℕ),
    fw.chair = 6 →
    fw.table = 9 →
    fw.stool = 2 →
    burn_rate = 5 →
    hours_warm = 34 →
    cf.tables = 6 →
    cf.stools = 4 →
    totalSticks fw cf = burn_rate * hours_warm →
    cf.chairs = 18 := by
  sorry

end mary_chopped_chairs_l1228_122823


namespace smallest_binary_palindrome_l1228_122864

/-- Checks if a natural number is a palindrome in the given base. -/
def is_palindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def to_base (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- The number 33 in decimal. -/
def target_number : ℕ := 33

theorem smallest_binary_palindrome :
  (is_palindrome target_number 2) ∧
  (∃ (b : ℕ), b > 2 ∧ is_palindrome target_number b) ∧
  (∀ (m : ℕ), m < target_number →
    ¬(is_palindrome m 2 ∧ (∃ (b : ℕ), b > 2 ∧ is_palindrome m b))) ∧
  (to_base target_number 2 = [1, 0, 0, 0, 0, 1]) :=
by sorry

end smallest_binary_palindrome_l1228_122864


namespace cubic_root_sum_cubes_l1228_122819

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (9 * a^3 - 27 * a + 54 = 0) →
  (9 * b^3 - 27 * b + 54 = 0) →
  (9 * c^3 - 27 * c + 54 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 18 := by
sorry

end cubic_root_sum_cubes_l1228_122819


namespace complementary_angles_difference_l1228_122878

theorem complementary_angles_difference (a b : Real) : 
  a + b = 90 →  -- angles are complementary
  a / b = 5 / 4 →  -- ratio of angles is 5:4
  (max a b - min a b) = 10 :=  -- positive difference is 10
by sorry

end complementary_angles_difference_l1228_122878


namespace clock_hands_opposite_l1228_122806

/-- Represents the number of minutes past 10:00 --/
def x : ℝ := 13

/-- The rate at which the minute hand moves (degrees per minute) --/
def minute_hand_rate : ℝ := 6

/-- The rate at which the hour hand moves (degrees per minute) --/
def hour_hand_rate : ℝ := 0.5

/-- The angle between the minute and hour hands when they are opposite --/
def opposite_angle : ℝ := 180

theorem clock_hands_opposite : 
  0 < x ∧ x < 60 ∧
  minute_hand_rate * (6 + x) + hour_hand_rate * (120 - x + 3) = opposite_angle :=
by sorry

end clock_hands_opposite_l1228_122806


namespace combined_rent_C_and_D_l1228_122800

-- Define the parameters for C and D
def oxen_C : ℕ := 15
def months_C : ℕ := 3
def rent_Z : ℕ := 100

def oxen_D : ℕ := 20
def months_D : ℕ := 6
def rent_W : ℕ := 120

-- Define the function to calculate rent
def calculate_rent (months : ℕ) (monthly_rent : ℕ) : ℕ :=
  months * monthly_rent

-- Theorem statement
theorem combined_rent_C_and_D :
  calculate_rent months_C rent_Z + calculate_rent months_D rent_W = 1020 := by
  sorry


end combined_rent_C_and_D_l1228_122800


namespace meaningful_fraction_iff_x_gt_three_l1228_122841

theorem meaningful_fraction_iff_x_gt_three (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 3)) ↔ x > 3 := by
sorry

end meaningful_fraction_iff_x_gt_three_l1228_122841


namespace two_digit_number_insertion_theorem_l1228_122812

theorem two_digit_number_insertion_theorem :
  ∃! (S : Finset Nat),
    (∀ n ∈ S, 10 ≤ n ∧ n < 100) ∧
    (∀ n ∉ S, ¬(10 ≤ n ∧ n < 100)) ∧
    (∀ n ∈ S,
      ∃ d : Nat,
      d < 10 ∧
      (100 * (n / 10) + 10 * d + (n % 10) = 9 * n)) ∧
    S.card = 4 := by
  sorry

end two_digit_number_insertion_theorem_l1228_122812


namespace square_plot_area_l1228_122805

/-- Proves that a square plot with given fence costs has an area of 144 square feet -/
theorem square_plot_area (cost_per_foot : ℝ) (total_cost : ℝ) : 
  cost_per_foot = 58 → total_cost = 2784 → 
  (total_cost / (4 * cost_per_foot))^2 = 144 := by
  sorry

end square_plot_area_l1228_122805


namespace lcm_of_4_6_15_l1228_122844

theorem lcm_of_4_6_15 : Nat.lcm (Nat.lcm 4 6) 15 = 60 := by
  sorry

end lcm_of_4_6_15_l1228_122844


namespace mikes_shortfall_l1228_122890

theorem mikes_shortfall (max_marks : ℕ) (mikes_score : ℕ) (passing_percentage : ℚ) : 
  max_marks = 750 → 
  mikes_score = 212 → 
  passing_percentage = 30 / 100 → 
  (↑max_marks * passing_percentage).floor - mikes_score = 13 := by
  sorry

end mikes_shortfall_l1228_122890


namespace sqrt_equation_solution_l1228_122835

theorem sqrt_equation_solution (z : ℝ) :
  (Real.sqrt 1.5 / Real.sqrt 0.81 + Real.sqrt z / Real.sqrt 0.49 = 3.0751133491652576) →
  z = 1.44 :=
by sorry

end sqrt_equation_solution_l1228_122835


namespace negation_of_existence_negation_of_quadratic_equation_l1228_122853

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ (∀ x, f x ≠ 0) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  apply negation_of_existence

end negation_of_existence_negation_of_quadratic_equation_l1228_122853


namespace x_equals_y_l1228_122836

theorem x_equals_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) : y = x := by
  sorry

end x_equals_y_l1228_122836


namespace b_current_age_b_current_age_proof_l1228_122807

theorem b_current_age : ℕ → ℕ → Prop :=
  fun a b =>
    (a = b + 15) →  -- A is 15 years older than B
    (a - 5 = 2 * (b - 5)) →  -- Five years ago, A's age was twice B's age
    (b = 20)  -- B's current age is 20

-- The proof is omitted
theorem b_current_age_proof : ∃ (a b : ℕ), b_current_age a b :=
  sorry

end b_current_age_b_current_age_proof_l1228_122807


namespace least_common_multiple_first_ten_l1228_122897

theorem least_common_multiple_first_ten : ∃ n : ℕ+, 
  (∀ k : ℕ+, k ≤ 10 → k ∣ n) ∧ 
  (∀ m : ℕ+, (∀ k : ℕ+, k ≤ 10 → k ∣ m) → n ≤ m) ∧
  n = 2520 := by
sorry

end least_common_multiple_first_ten_l1228_122897


namespace right_triangle_area_l1228_122857

theorem right_triangle_area (hypotenuse base : ℝ) (h1 : hypotenuse = 15) (h2 : base = 9) :
  let height : ℝ := Real.sqrt (hypotenuse^2 - base^2)
  let area : ℝ := (base * height) / 2
  area = 54 := by sorry

end right_triangle_area_l1228_122857


namespace count_pairs_sum_squares_less_than_50_l1228_122898

theorem count_pairs_sum_squares_less_than_50 :
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + p.2^2 < 50)
    (Finset.product (Finset.range 50) (Finset.range 50))).card = 32 :=
by sorry

end count_pairs_sum_squares_less_than_50_l1228_122898


namespace original_calculation_l1228_122867

theorem original_calculation (x : ℚ) (h : ((x * 3) + 14) * 2 = 946) : ((x / 3) + 14) * 2 = 130 := by
  sorry

end original_calculation_l1228_122867


namespace stratified_sampling_type_D_l1228_122834

/-- Calculates the number of units to be selected from a specific product type in stratified sampling -/
def stratifiedSampleSize (totalProduction : ℕ) (typeProduction : ℕ) (totalSample : ℕ) : ℕ :=
  (typeProduction * totalSample) / totalProduction

/-- The problem statement -/
theorem stratified_sampling_type_D :
  let totalProduction : ℕ := 100 + 200 + 300 + 400
  let typeDProduction : ℕ := 400
  let totalSample : ℕ := 50
  stratifiedSampleSize totalProduction typeDProduction totalSample = 20 := by
  sorry


end stratified_sampling_type_D_l1228_122834


namespace find_number_l1228_122825

theorem find_number : ∃ x : ℝ, (0.4 * x = 0.75 * 100 + 50) ∧ (x = 312.5) := by
  sorry

end find_number_l1228_122825


namespace rabbits_to_add_correct_rabbits_to_add_l1228_122847

theorem rabbits_to_add (initial_rabbits : ℕ) (park_rabbits : ℕ) : ℕ :=
  let final_rabbits := park_rabbits / 3
  final_rabbits - initial_rabbits

theorem correct_rabbits_to_add :
  rabbits_to_add 13 60 = 7 := by sorry

end rabbits_to_add_correct_rabbits_to_add_l1228_122847


namespace tan_fifteen_ratio_equals_sqrt_three_l1228_122837

theorem tan_fifteen_ratio_equals_sqrt_three : 
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end tan_fifteen_ratio_equals_sqrt_three_l1228_122837


namespace arithmetic_calculation_l1228_122866

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 36 / 3 + 48 / 4 = 71 := by
  sorry

end arithmetic_calculation_l1228_122866


namespace sequence_formula_l1228_122856

/-- A sequence where S_n = n^2 * a_n for n ≥ 2, and a_1 = 1 -/
def sequence_a (n : ℕ) : ℚ := sorry

/-- Sum of the first n terms of the sequence -/
def S (n : ℕ) : ℚ := sorry

theorem sequence_formula :
  ∀ n : ℕ, n ≥ 1 →
  (∀ k : ℕ, k ≥ 2 → S k = k^2 * sequence_a k) →
  sequence_a 1 = 1 →
  sequence_a n = 1 / (n + 1) :=
sorry

end sequence_formula_l1228_122856


namespace quadratic_inequality_l1228_122877

-- Define the quadratic function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Define the linear function g
def g (x : ℝ) : ℝ := x - 1

-- Theorem statement
theorem quadratic_inequality (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 1 < x ∧ x < 2) →
  (a = 3 ∧ b = 2) ∧
  (∀ c : ℝ,
    (c > -1 → ∀ x, f a b x > c * g x ↔ x > c + 2 ∨ x < 1) ∧
    (c < -1 → ∀ x, f a b x > c * g x ↔ x > 1 ∨ x < c + 2) ∧
    (c = -1 → ∀ x, f a b x > c * g x ↔ x ≠ 1)) :=
by sorry


end quadratic_inequality_l1228_122877


namespace debby_water_bottles_l1228_122884

/-- The number of bottles Debby drinks per day -/
def bottles_per_day : ℕ := 5

/-- The number of days the water would last -/
def days_lasting : ℕ := 71

/-- The total number of bottles Debby bought -/
def total_bottles : ℕ := bottles_per_day * days_lasting

theorem debby_water_bottles : total_bottles = 355 := by
  sorry

end debby_water_bottles_l1228_122884


namespace min_value_sum_reciprocals_l1228_122886

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) :
  (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ≥ 9/2 := by
  sorry

end min_value_sum_reciprocals_l1228_122886


namespace unpainted_area_crossed_boards_l1228_122880

/-- The area of the unpainted region when two boards cross at 45 degrees -/
theorem unpainted_area_crossed_boards (width1 width2 : ℝ) (angle : ℝ) :
  width1 = 5 →
  width2 = 7 →
  angle = π / 4 →
  let projected_length := width2
  let overlap_height := width2 * Real.cos angle
  let unpainted_area := projected_length * overlap_height
  unpainted_area = 49 * Real.sqrt 2 / 2 := by
  sorry

end unpainted_area_crossed_boards_l1228_122880


namespace four_students_three_events_outcomes_l1228_122808

/-- The number of possible outcomes for champions in a competition --/
def championOutcomes (students : ℕ) (events : ℕ) : ℕ :=
  students ^ events

/-- Theorem: Given 4 students and 3 events, the number of possible outcomes for champions is 64 --/
theorem four_students_three_events_outcomes :
  championOutcomes 4 3 = 64 := by
  sorry

end four_students_three_events_outcomes_l1228_122808


namespace floor_sqrt_120_l1228_122811

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end floor_sqrt_120_l1228_122811


namespace zachary_crunches_l1228_122891

/-- Proves that Zachary did 14 crunches given the conditions -/
theorem zachary_crunches : 
  ∀ (zachary_pushups zachary_total : ℕ),
  zachary_pushups = 53 →
  zachary_total = 67 →
  zachary_total - zachary_pushups = 14 :=
by
  sorry

end zachary_crunches_l1228_122891


namespace complex_modulus_example_l1228_122869

theorem complex_modulus_example : 
  let z : ℂ := 1 - 2*I
  Complex.abs z = Real.sqrt 5 := by sorry

end complex_modulus_example_l1228_122869


namespace triangle_area_is_one_l1228_122842

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the area of a specific triangle -/
theorem triangle_area_is_one (t : Triangle) 
  (h1 : (Real.cos t.B / t.b) + (Real.cos t.C / t.c) = (Real.sin t.A / (2 * Real.sin t.C)))
  (h2 : Real.sqrt 3 * t.b * Real.cos t.C = (2 * t.a - Real.sqrt 3 * t.c) * Real.cos t.B)
  (h3 : ∃ (r : ℝ), Real.sin t.A = r * Real.sin t.B ∧ Real.sin t.B = r * Real.sin t.C) :
  (1/2) * t.a * t.c * Real.sin t.B = 1 := by
  sorry

#check triangle_area_is_one

end triangle_area_is_one_l1228_122842


namespace fraction_inequality_implies_numerator_inequality_l1228_122846

theorem fraction_inequality_implies_numerator_inequality
  (a b c : ℝ) (hc : c ≠ 0) :
  a / c^2 > b / c^2 → a > b := by
  sorry

end fraction_inequality_implies_numerator_inequality_l1228_122846


namespace system_solution_l1228_122817

theorem system_solution : 
  ∀ x y : ℝ, 
    x^2 + 3*x*y = 18 ∧ x*y + 3*y^2 = 6 → 
      (x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = -1) :=
by
  sorry

end system_solution_l1228_122817


namespace square_of_integer_root_l1228_122832

theorem square_of_integer_root (n : ℕ) : 
  ∃ (m : ℤ), (2 : ℝ) + 2 * Real.sqrt (28 * (n^2 : ℝ) + 1) = m → 
  ∃ (k : ℤ), m = k^2 := by
sorry

end square_of_integer_root_l1228_122832


namespace function_1_extrema_function_2_extrema_l1228_122861

-- Function 1
theorem function_1_extrema :
  (∀ x : ℝ, 2 * Real.sin x - 3 ≤ -1) ∧
  (∃ x : ℝ, 2 * Real.sin x - 3 = -1) ∧
  (∀ x : ℝ, 2 * Real.sin x - 3 ≥ -5) ∧
  (∃ x : ℝ, 2 * Real.sin x - 3 = -5) :=
sorry

-- Function 2
theorem function_2_extrema :
  (∀ x : ℝ, 7/4 + Real.sin x - (Real.sin x)^2 ≤ 2) ∧
  (∃ x : ℝ, 7/4 + Real.sin x - (Real.sin x)^2 = 2) ∧
  (∀ x : ℝ, 7/4 + Real.sin x - (Real.sin x)^2 ≥ -1/4) ∧
  (∃ x : ℝ, 7/4 + Real.sin x - (Real.sin x)^2 = -1/4) :=
sorry

end function_1_extrema_function_2_extrema_l1228_122861


namespace alien_invasion_characteristics_l1228_122818

-- Define the characteristics of an alien species invasion
structure AlienInvasion where
  j_shaped_growth : Bool
  unrestricted_growth : Bool
  threatens_biodiversity : Bool
  eliminated_if_unadapted : Bool

-- Define the correct characteristics of an alien invasion
def correct_invasion : AlienInvasion :=
  { j_shaped_growth := true,
    unrestricted_growth := false,
    threatens_biodiversity := true,
    eliminated_if_unadapted := true }

-- Theorem: The correct characteristics of an alien invasion are as defined
theorem alien_invasion_characteristics :
  ∃ (invasion : AlienInvasion),
    invasion.j_shaped_growth ∧
    ¬invasion.unrestricted_growth ∧
    invasion.threatens_biodiversity ∧
    invasion.eliminated_if_unadapted :=
by
  sorry


end alien_invasion_characteristics_l1228_122818


namespace number_exists_l1228_122851

theorem number_exists : ∃ x : ℝ, 0.75 * x = 0.3 * 1000 + 250 := by
  sorry

end number_exists_l1228_122851


namespace operation_result_l1228_122816

-- Define a type for the allowed operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

theorem operation_result 
  (diamond circ : Operation) 
  (h : (apply_op diamond 20 4) / (apply_op circ 12 4) = 2) :
  (apply_op diamond 9 3) / (apply_op circ 15 5) = 27 / 20 :=
by sorry

end operation_result_l1228_122816


namespace work_completion_time_l1228_122871

theorem work_completion_time (aarti_rate ramesh_rate : ℚ) 
  (h1 : aarti_rate = 1 / 6)
  (h2 : ramesh_rate = 1 / 8)
  (h3 : (aarti_rate + ramesh_rate) * 3 = 1) :
  3 = 3 := by sorry

end work_completion_time_l1228_122871


namespace intersection_of_M_and_N_l1228_122810

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end intersection_of_M_and_N_l1228_122810


namespace water_tank_capacity_l1228_122802

theorem water_tank_capacity : ∀ x : ℚ, 
  (3/4 : ℚ) * x - (1/3 : ℚ) * x = 15 → x = 36 :=
by
  sorry

end water_tank_capacity_l1228_122802


namespace no_real_solutions_for_quadratic_inequality_l1228_122845

theorem no_real_solutions_for_quadratic_inequality :
  ¬ ∃ x : ℝ, 3 * x^2 + 9 * x ≤ -12 := by
sorry

end no_real_solutions_for_quadratic_inequality_l1228_122845


namespace min_sum_given_product_l1228_122840

theorem min_sum_given_product (a b : ℝ) : 
  a > 0 → b > 0 → a + b = a * b → (∀ x y : ℝ, x > 0 → y > 0 → x + y = x * y → a + b ≤ x + y) → a + b = 4 := by
  sorry

end min_sum_given_product_l1228_122840


namespace stratified_sampling_theorem_l1228_122888

/-- The number of different arrangements for selecting 4 students (1 girl and 3 boys) 
    from 8 students (6 boys and 2 girls) by stratified sampling based on gender, 
    with a girl as the first runner. -/
def stratifiedSamplingArrangements : ℕ := sorry

/-- The total number of students -/
def totalStudents : ℕ := 8

/-- The number of boys -/
def numBoys : ℕ := 6

/-- The number of girls -/
def numGirls : ℕ := 2

/-- The number of students to be selected -/
def selectedStudents : ℕ := 4

/-- The number of boys to be selected -/
def selectedBoys : ℕ := 3

/-- The number of girls to be selected -/
def selectedGirls : ℕ := 1

theorem stratified_sampling_theorem : 
  stratifiedSamplingArrangements = 240 :=
sorry

end stratified_sampling_theorem_l1228_122888


namespace hot_dog_sales_l1228_122833

theorem hot_dog_sales (total : ℕ) (first_innings : ℕ) (left_unsold : ℕ) :
  total = 91 →
  first_innings = 19 →
  left_unsold = 45 →
  total - (first_innings + left_unsold) = 27 := by
  sorry

end hot_dog_sales_l1228_122833


namespace least_common_denominator_l1228_122848

theorem least_common_denominator : 
  let denominators : List Nat := [3, 4, 5, 6, 8, 9, 10]
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 3 4) 5) 6) 8) 9) 10 = 360 := by
  sorry

end least_common_denominator_l1228_122848


namespace smallest_n_for_four_sum_divisible_by_four_l1228_122803

theorem smallest_n_for_four_sum_divisible_by_four :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a + b + c + d) % 4 = 0) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℤ), T.card = m ∧
    ∀ (a b c d : ℤ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    (a + b + c + d) % 4 ≠ 0) ∧
  n = 7 :=
by sorry

end smallest_n_for_four_sum_divisible_by_four_l1228_122803


namespace quadratic_equation_properties_l1228_122821

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - m*x - 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (f (Real.sqrt 2) = 0 → f (-Real.sqrt 2 / 2) = 0 ∧ m = Real.sqrt 2 / 2) :=
by sorry

end quadratic_equation_properties_l1228_122821


namespace ac_length_l1228_122883

/-- An isosceles trapezoid with specific measurements -/
structure IsoscelesTrapezoid where
  /-- Length of base AB -/
  ab : ℝ
  /-- Length of top side CD -/
  cd : ℝ
  /-- Length of leg AD (equal to BC) -/
  ad : ℝ
  /-- Constraint that AB > CD -/
  h_ab_gt_cd : ab > cd

/-- Theorem: In the given isosceles trapezoid, AC = 17 -/
theorem ac_length (t : IsoscelesTrapezoid) 
  (h_ab : t.ab = 21)
  (h_cd : t.cd = 9)
  (h_ad : t.ad = 10) : 
  Real.sqrt ((21 - 9) ^ 2 / 4 + 8 ^ 2) = 17 := by
  sorry


end ac_length_l1228_122883


namespace parabola_line_intersection_l1228_122850

/-- Parabola P with equation y = x^2 + 3x + 1 -/
def P : ℝ → ℝ := λ x => x^2 + 3*x + 1

/-- Point Q -/
def Q : ℝ × ℝ := (10, 50)

/-- Line through Q with slope m -/
def line (m : ℝ) : ℝ → ℝ := λ x => m*(x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line m x

/-- Main theorem -/
theorem parabola_line_intersection :
  ∃! (r s : ℝ), (∀ m, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 46 := by sorry

end parabola_line_intersection_l1228_122850


namespace complex_real_condition_l1228_122828

theorem complex_real_condition (a : ℝ) : 
  let z : ℂ := (a - 3 : ℂ) + (a^2 - 2*a - 3 : ℂ) * Complex.I
  (z.im = 0) → (a = 3 ∨ a = -1) := by
sorry

end complex_real_condition_l1228_122828


namespace cosine_symmetry_center_l1228_122813

/-- The symmetry center of the cosine function with a phase shift --/
theorem cosine_symmetry_center (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.cos (2 * x - π / 4)
  ∃ c : ℝ × ℝ, c = (π / 8, 0) ∧ 
    (∀ x : ℝ, f (c.1 + x) = f (c.1 - x)) :=
by sorry

end cosine_symmetry_center_l1228_122813


namespace city_population_theorem_l1228_122885

def city_population (initial_population immigration emigration pregnancy_rate twin_rate : ℕ) : ℕ :=
  let population_after_migration := initial_population + immigration - emigration
  let pregnancies := population_after_migration / 8
  let twin_pregnancies := pregnancies / 4
  let single_pregnancies := pregnancies - twin_pregnancies
  let births := single_pregnancies + 2 * twin_pregnancies
  population_after_migration + births

theorem city_population_theorem :
  city_population 300000 50000 30000 8 4 = 370000 := by
  sorry

end city_population_theorem_l1228_122885


namespace geometry_propositions_l1228_122838

structure Geometry3D where
  Line : Type
  Plane : Type
  parallel : Line → Plane → Prop
  perpendicular : Line → Plane → Prop
  plane_parallel : Plane → Plane → Prop
  plane_perpendicular : Plane → Plane → Prop

variable (G : Geometry3D)

theorem geometry_propositions 
  (l : G.Line) (α β : G.Plane) (h_diff : α ≠ β) :
  (∃ l α β, G.parallel l α ∧ G.parallel l β ∧ ¬ G.plane_parallel α β) ∧
  (∀ l α β, G.perpendicular l α ∧ G.perpendicular l β → G.plane_parallel α β) ∧
  (∃ l α β, G.perpendicular l α ∧ G.parallel l β ∧ ¬ G.plane_parallel α β) ∧
  (∃ l α β, G.plane_perpendicular α β ∧ G.parallel l α ∧ ¬ G.perpendicular l β) :=
by sorry

end geometry_propositions_l1228_122838


namespace probability_rain_july_approx_l1228_122814

/-- The probability of rain on at most 1 day in July, given the daily rain probability and number of days. -/
def probability_rain_at_most_one_day (daily_prob : ℝ) (num_days : ℕ) : ℝ :=
  (1 - daily_prob) ^ num_days + num_days * daily_prob * (1 - daily_prob) ^ (num_days - 1)

/-- Theorem stating that the probability of rain on at most 1 day in July is approximately 0.271. -/
theorem probability_rain_july_approx : 
  ∃ ε > 0, ε < 0.001 ∧ 
  |probability_rain_at_most_one_day (1/20) 31 - 0.271| < ε :=
sorry

end probability_rain_july_approx_l1228_122814


namespace boat_speed_in_still_water_l1228_122875

/-- Proves that a boat traveling 45 miles upstream in 5 hours and 45 miles downstream in 3 hours has a speed of 12 mph in still water -/
theorem boat_speed_in_still_water : 
  ∀ (upstream_speed downstream_speed : ℝ),
  upstream_speed = 45 / 5 →
  downstream_speed = 45 / 3 →
  ∃ (boat_speed current_speed : ℝ),
  boat_speed - current_speed = upstream_speed ∧
  boat_speed + current_speed = downstream_speed ∧
  boat_speed = 12 := by
sorry

end boat_speed_in_still_water_l1228_122875


namespace three_digit_numbers_sum_divisibility_l1228_122860

theorem three_digit_numbers_sum_divisibility :
  ∃ (a b c d : ℕ),
    100 ≤ a ∧ a < 1000 ∧
    100 ≤ b ∧ b < 1000 ∧
    100 ≤ c ∧ c < 1000 ∧
    100 ≤ d ∧ d < 1000 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∃ (j : ℕ), a / 100 = j ∧ b / 100 = j ∧ c / 100 = j ∧ d / 100 = j) ∧
    (∃ (s : ℕ), s = a + b + c + d ∧ s % a = 0 ∧ s % b = 0 ∧ s % c = 0) ∧
    a = 108 ∧ b = 135 ∧ c = 180 ∧ d = 117 :=
by sorry

end three_digit_numbers_sum_divisibility_l1228_122860


namespace max_area_inscribed_triangle_l1228_122858

/-- The ellipse in which the triangle is inscribed -/
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

/-- A point on the ellipse -/
structure EllipsePoint where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

/-- The triangle inscribed in the ellipse -/
structure InscribedTriangle where
  A : EllipsePoint
  B : EllipsePoint
  C : EllipsePoint

/-- The condition that line segment AB passes through point P(1,0) -/
def passes_through_P (t : InscribedTriangle) : Prop :=
  ∃ k : ℝ, t.A.x + k * (t.B.x - t.A.x) = 1 ∧ t.A.y + k * (t.B.y - t.A.y) = 0

/-- The area of the triangle -/
noncomputable def triangle_area (t : InscribedTriangle) : ℝ :=
  abs ((t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y)) / 2

/-- The theorem to be proved -/
theorem max_area_inscribed_triangle :
  ∃ (t : InscribedTriangle), passes_through_P t ∧
    (∀ (t' : InscribedTriangle), passes_through_P t' → triangle_area t' ≤ triangle_area t) ∧
    triangle_area t = 16 * Real.sqrt 2 / 3 := by
  sorry

end max_area_inscribed_triangle_l1228_122858


namespace f_at_2_l1228_122831

def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem f_at_2 : f 2 = 123 := by
  sorry

end f_at_2_l1228_122831


namespace seventh_power_of_complex_l1228_122815

theorem seventh_power_of_complex (z : ℂ) : 
  z = (Real.sqrt 3 + Complex.I) / 2 → z^7 = -Real.sqrt 3 / 2 - Complex.I / 2 := by
  sorry

end seventh_power_of_complex_l1228_122815


namespace fold_crease_length_l1228_122822

/-- Represents a rectangular sheet of paper -/
structure Sheet :=
  (length : ℝ)
  (width : ℝ)

/-- Represents the crease formed by folding the sheet -/
def crease_length (s : Sheet) : ℝ :=
  sorry

/-- The theorem stating the length of the crease -/
theorem fold_crease_length (s : Sheet) 
  (h1 : s.length = 8) 
  (h2 : s.width = 6) : 
  crease_length s = 7.5 :=
sorry

end fold_crease_length_l1228_122822


namespace boat_downstream_distance_l1228_122889

/-- The distance traveled by a boat downstream -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Theorem: A boat with speed 13 km/hr in still water, traveling downstream
    in a stream with speed 4 km/hr for 4 hours, covers a distance of 68 km -/
theorem boat_downstream_distance :
  distance_downstream 13 4 4 = 68 := by
  sorry

end boat_downstream_distance_l1228_122889


namespace real_roots_of_polynomial_l1228_122824

theorem real_roots_of_polynomial (x : ℝ) :
  x^4 - 2*x^3 - x + 2 = 0 ↔ x = 1 ∨ x = 2 :=
sorry

end real_roots_of_polynomial_l1228_122824


namespace square_area_proof_l1228_122843

theorem square_area_proof (x : ℚ) : 
  (5 * x - 20 = 25 - 2 * x) → 
  ((5 * x - 20)^2 : ℚ) = 7225 / 49 := by
sorry

end square_area_proof_l1228_122843


namespace dodecagon_diagonals_l1228_122863

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: A regular dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l1228_122863


namespace fourth_grade_students_l1228_122826

/-- The number of students in fourth grade at the end of the year -/
def final_students (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem: Given the initial number of students, the number of students who left,
    and the number of new students, prove that the final number of students is 47 -/
theorem fourth_grade_students :
  final_students 11 6 42 = 47 := by
  sorry

end fourth_grade_students_l1228_122826


namespace number_of_values_l1228_122865

theorem number_of_values (initial_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) (correct_mean : ℚ) :
  initial_mean = 180 →
  incorrect_value = 135 →
  correct_value = 155 →
  correct_mean = 180 + 2/3 →
  ∃ n : ℕ, 
    n * initial_mean = n * correct_mean - (correct_value - incorrect_value) ∧
    n = 60 :=
by sorry

end number_of_values_l1228_122865


namespace cricket_players_count_l1228_122893

/-- The number of cricket players in a game, given the numbers of other players and the total. -/
theorem cricket_players_count 
  (hockey_players : ℕ) 
  (football_players : ℕ) 
  (softball_players : ℕ) 
  (total_players : ℕ) 
  (h1 : hockey_players = 12)
  (h2 : football_players = 16)
  (h3 : softball_players = 13)
  (h4 : total_players = 51)
  (h5 : total_players = hockey_players + football_players + softball_players + cricket_players) :
  cricket_players = 10 := by
  sorry

end cricket_players_count_l1228_122893


namespace no_bribed_judges_probability_l1228_122829

def total_judges : ℕ := 14
def valid_scores : ℕ := 7
def bribed_judges : ℕ := 2

def probability_no_bribed_judges : ℚ := 3/13

theorem no_bribed_judges_probability :
  (Nat.choose (total_judges - bribed_judges) valid_scores * Nat.choose bribed_judges 0) /
  Nat.choose total_judges valid_scores = probability_no_bribed_judges := by
  sorry

end no_bribed_judges_probability_l1228_122829


namespace rhombus_perimeter_l1228_122809

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 16 feet is 16√13 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 16 * Real.sqrt 13 := by
  sorry

end rhombus_perimeter_l1228_122809


namespace factorization_problems_l1228_122870

theorem factorization_problems (x : ℝ) : 
  (2 * x^2 - 8 = 2 * (x + 2) * (x - 2)) ∧ 
  (2 * x^2 + 2 * x + (1/2) = 2 * (x + 1/2)^2) := by
  sorry

end factorization_problems_l1228_122870


namespace bowling_ball_weights_l1228_122849

/-- The weight of a single canoe in pounds -/
def canoe_weight : ℕ := 36

/-- The number of bowling balls that weigh the same as the canoes -/
def num_bowling_balls : ℕ := 9

/-- The number of canoes that weigh the same as the bowling balls -/
def num_canoes : ℕ := 4

/-- The weight of a single bowling ball in pounds -/
def bowling_ball_weight : ℕ := canoe_weight * num_canoes / num_bowling_balls

/-- The total weight of five bowling balls in pounds -/
def five_bowling_balls_weight : ℕ := bowling_ball_weight * 5

theorem bowling_ball_weights :
  bowling_ball_weight = 16 ∧ five_bowling_balls_weight = 80 := by
  sorry

end bowling_ball_weights_l1228_122849
