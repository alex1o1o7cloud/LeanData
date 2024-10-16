import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2905_290588

theorem equation_solution : ∃ x : ℚ, (1 / 3 + 1 / x = 2 / 3) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2905_290588


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2905_290507

/-- Given that i is the imaginary unit, prove that (1+2i)/(1+i) = (3+i)/2 -/
theorem complex_fraction_equality : (1 + 2 * Complex.I) / (1 + Complex.I) = (3 + Complex.I) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2905_290507


namespace NUMINAMATH_CALUDE_pure_imaginary_iff_x_eq_one_l2905_290578

theorem pure_imaginary_iff_x_eq_one (x : ℝ) :
  x = 1 ↔ (Complex.mk (x^2 - 1) (x + 1)).im ≠ 0 ∧ (Complex.mk (x^2 - 1) (x + 1)).re = 0 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_iff_x_eq_one_l2905_290578


namespace NUMINAMATH_CALUDE_tg_ctg_equation_solution_l2905_290570

theorem tg_ctg_equation_solution (x : ℝ) :
  (∀ n : ℤ, x ≠ (n : ℝ) * π / 2) →
  (Real.tan x ^ 4 + (1 / Real.tan x) ^ 4 = (82 / 9) * (Real.tan x * Real.tan (2 * x) + 1) * Real.cos (2 * x)) ↔
  ∃ k : ℤ, x = π / 6 * ((3 * k : ℝ) + 1) ∨ x = π / 6 * ((3 * k : ℝ) - 1) :=
by sorry

end NUMINAMATH_CALUDE_tg_ctg_equation_solution_l2905_290570


namespace NUMINAMATH_CALUDE_polynomial_inequality_l2905_290543

theorem polynomial_inequality (a b c d : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |3 * a * x^2 + 2 * b * x + c| ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l2905_290543


namespace NUMINAMATH_CALUDE_problem_part1_problem_part2_l2905_290533

-- Part 1
theorem problem_part1 : (-2)^3 + |(-3)| - Real.tan (π/4) = -6 := by sorry

-- Part 2
theorem problem_part2 (a : ℝ) : (a + 2)^2 - a*(a - 4) = 8*a + 4 := by sorry

end NUMINAMATH_CALUDE_problem_part1_problem_part2_l2905_290533


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l2905_290554

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℤ) :
  arithmetic_sequence a → a 4 = -4 → a 8 = 4 → a 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l2905_290554


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l2905_290512

/-- Represents a triangle with an area and a side length -/
structure Triangle where
  area : ℝ
  side : ℝ

/-- Proves that given two similar triangles with specific properties, 
    the corresponding side of the larger triangle is 12 feet -/
theorem similar_triangles_side_length 
  (t1 t2 : Triangle) 
  (h_area_diff : t1.area - t2.area = 32)
  (h_area_ratio : t1.area / t2.area = 9)
  (h_smaller_area_int : ∃ n : ℕ, t2.area = n)
  (h_smaller_side : t2.side = 4)
  (h_similar : ∃ k : ℝ, t1.side = k * t2.side ∧ t1.area = k^2 * t2.area) :
  t1.side = 12 := by
  sorry

#check similar_triangles_side_length

end NUMINAMATH_CALUDE_similar_triangles_side_length_l2905_290512


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2905_290504

theorem smallest_four_digit_divisible_by_53 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 1007 → ¬(53 ∣ n)) ∧ 53 ∣ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2905_290504


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l2905_290575

/-- Represents the speed of a boat in still water -/
def boat_speed : ℝ := 15

/-- Represents the speed of the stream -/
def stream_speed : ℝ := 10

/-- Represents the downstream distance traveled -/
def downstream_distance : ℝ := 100

/-- Represents the upstream distance traveled -/
def upstream_distance : ℝ := 75

/-- Represents the time taken for downstream travel -/
def downstream_time : ℝ := 4

/-- Represents the time taken for upstream travel -/
def upstream_time : ℝ := 15

theorem stream_speed_calculation :
  (downstream_distance / downstream_time = boat_speed + stream_speed) ∧
  (upstream_distance / upstream_time = boat_speed - stream_speed) →
  stream_speed = 10 := by sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l2905_290575


namespace NUMINAMATH_CALUDE_base_six_addition_l2905_290501

/-- Given a base-6 addition 4AB₆ + 41₆ = 53A₆, prove that A + B = 9 in base 10 -/
theorem base_six_addition (A B : ℕ) : 
  (4 * 6^2 + A * 6 + B) + (4 * 6 + 1) = 5 * 6^2 + 3 * 6 + A → A + B = 9 :=
by sorry

end NUMINAMATH_CALUDE_base_six_addition_l2905_290501


namespace NUMINAMATH_CALUDE_discriminant_greater_than_four_l2905_290582

theorem discriminant_greater_than_four (p q : ℝ) 
  (h1 : 999^2 + p * 999 + q < 0) 
  (h2 : 1001^2 + p * 1001 + q < 0) : 
  p^2 - 4*q > 4 := by
sorry

end NUMINAMATH_CALUDE_discriminant_greater_than_four_l2905_290582


namespace NUMINAMATH_CALUDE_clarence_oranges_l2905_290532

/-- The number of oranges Clarence has initially -/
def initial_oranges : ℕ := 5

/-- The number of oranges Clarence receives from Joyce -/
def oranges_from_joyce : ℕ := 3

/-- The total number of oranges Clarence has -/
def total_oranges : ℕ := initial_oranges + oranges_from_joyce

theorem clarence_oranges : total_oranges = 8 := by
  sorry

end NUMINAMATH_CALUDE_clarence_oranges_l2905_290532


namespace NUMINAMATH_CALUDE_intersection_orthogonal_l2905_290522

/-- The ellipse E with equation x²/8 + y²/4 = 1 -/
def E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}

/-- The line L with equation y = √5*x + 4 -/
def L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = Real.sqrt 5 * p.1 + 4}

/-- The intersection points of E and L -/
def intersection := E ∩ L

/-- Theorem: If A and B are the intersection points of E and L, then OA ⊥ OB -/
theorem intersection_orthogonal (A B : ℝ × ℝ) 
  (hA : A ∈ intersection) (hB : B ∈ intersection) (hAB : A ≠ B) :
  (A.1 * B.1 + A.2 * B.2 = 0) := by sorry

end NUMINAMATH_CALUDE_intersection_orthogonal_l2905_290522


namespace NUMINAMATH_CALUDE_sphere_triangle_distance_l2905_290574

theorem sphere_triangle_distance (r : ℝ) (a b : ℝ) (hr : r = 13) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let d := Real.sqrt (r^2 - (c/2)^2)
  d = 12 := by sorry

end NUMINAMATH_CALUDE_sphere_triangle_distance_l2905_290574


namespace NUMINAMATH_CALUDE_kaylee_biscuits_l2905_290599

def biscuit_problem (total_needed : ℕ) (lemon_sold : ℕ) (chocolate_sold : ℕ) (oatmeal_sold : ℕ) : ℕ :=
  total_needed - (lemon_sold + chocolate_sold + oatmeal_sold)

theorem kaylee_biscuits :
  biscuit_problem 33 12 5 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_kaylee_biscuits_l2905_290599


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2905_290536

theorem inscribed_cube_surface_area :
  ∀ (outer_cube_surface_area : ℝ) (inner_cube_surface_area : ℝ),
    outer_cube_surface_area = 54 →
    (∃ (outer_cube_side : ℝ) (sphere_diameter : ℝ) (inner_cube_diagonal : ℝ) (inner_cube_side : ℝ),
      outer_cube_surface_area = 6 * outer_cube_side^2 ∧
      sphere_diameter = outer_cube_side ∧
      inner_cube_diagonal = sphere_diameter ∧
      inner_cube_diagonal = inner_cube_side * Real.sqrt 3 ∧
      inner_cube_surface_area = 6 * inner_cube_side^2) →
    inner_cube_surface_area = 18 :=
by
  sorry


end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2905_290536


namespace NUMINAMATH_CALUDE_spinner_probability_l2905_290559

theorem spinner_probability (p_A p_B p_C p_DE : ℚ) : 
  p_A = 1/3 →
  p_B = 1/6 →
  p_C = p_DE →
  p_A + p_B + p_C + p_DE = 1 →
  p_C = 1/4 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l2905_290559


namespace NUMINAMATH_CALUDE_cubic_inequality_l2905_290538

theorem cubic_inequality (x : ℝ) :
  x ≥ 0 → (x^3 - 9*x^2 - 16*x > 0 ↔ x > 16) := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2905_290538


namespace NUMINAMATH_CALUDE_male_athletes_to_sample_l2905_290558

theorem male_athletes_to_sample (total_athletes : ℕ) (female_athletes : ℕ) (selection_prob : ℚ) :
  total_athletes = 98 →
  female_athletes = 42 →
  selection_prob = 2 / 7 →
  (total_athletes - female_athletes) * selection_prob = 16 := by
  sorry

end NUMINAMATH_CALUDE_male_athletes_to_sample_l2905_290558


namespace NUMINAMATH_CALUDE_dark_box_probability_l2905_290505

theorem dark_box_probability (a : ℕ) : 
  a > 0 → 
  (3 : ℝ) / a = (1 : ℝ) / 4 → 
  a = 12 := by
sorry

end NUMINAMATH_CALUDE_dark_box_probability_l2905_290505


namespace NUMINAMATH_CALUDE_total_damage_cost_l2905_290572

/-- The cost of damages caused by Jack --/
def cost_of_damages (tire_cost : ℕ) (num_tires : ℕ) (window_cost : ℕ) : ℕ :=
  tire_cost * num_tires + window_cost

/-- Theorem stating the total cost of damages --/
theorem total_damage_cost :
  cost_of_damages 250 3 700 = 1450 := by
  sorry

end NUMINAMATH_CALUDE_total_damage_cost_l2905_290572


namespace NUMINAMATH_CALUDE_watch_loss_percentage_l2905_290525

/-- Proves that the loss percentage is 5% when a watch is sold at a loss for 1140,
    given that selling it for 1260 would result in a 5% profit. -/
theorem watch_loss_percentage
  (cost_price : ℝ)
  (loss_price : ℝ)
  (profit_price : ℝ)
  (profit_percentage : ℝ)
  (h1 : loss_price = 1140)
  (h2 : profit_price = 1260)
  (h3 : profit_percentage = 0.05)
  (h4 : profit_price = cost_price * (1 + profit_percentage))
  (h5 : loss_price < cost_price) :
  (cost_price - loss_price) / cost_price = 0.05 := by
sorry

end NUMINAMATH_CALUDE_watch_loss_percentage_l2905_290525


namespace NUMINAMATH_CALUDE_solve_for_b_l2905_290548

theorem solve_for_b (x b : ℝ) : 
  (10 * x + b) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3 →
  x = 0.3 →
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_b_l2905_290548


namespace NUMINAMATH_CALUDE_pet_shop_solution_l2905_290553

/-- Represents the pet shop inventory --/
structure PetShop where
  kittens : ℕ
  hamsters : ℕ
  birds : ℕ
  puppies : ℕ

/-- The initial state of the pet shop --/
def initial_state : PetShop :=
  { kittens := 45,
    hamsters := 30,
    birds := 60,
    puppies := 15 }

/-- The final state of the pet shop after changes --/
def final_state : PetShop :=
  { kittens := initial_state.kittens,
    hamsters := initial_state.hamsters,
    birds := initial_state.birds + 10,
    puppies := initial_state.puppies - 5 }

/-- Theorem stating the correctness of the solution --/
theorem pet_shop_solution :
  (initial_state.kittens + initial_state.hamsters + initial_state.birds + initial_state.puppies = 150) ∧
  (3 * initial_state.hamsters = 2 * initial_state.kittens) ∧
  (initial_state.birds = initial_state.hamsters + 30) ∧
  (4 * initial_state.puppies = initial_state.birds) ∧
  (final_state.kittens + final_state.hamsters + final_state.birds + final_state.puppies = 155) ∧
  (final_state.kittens = 45) ∧
  (final_state.hamsters = 30) ∧
  (final_state.birds = 70) ∧
  (final_state.puppies = 10) := by
  sorry


end NUMINAMATH_CALUDE_pet_shop_solution_l2905_290553


namespace NUMINAMATH_CALUDE_hans_deposit_is_101_l2905_290542

/-- Calculates the deposit for a restaurant reservation --/
def calculate_deposit (num_adults num_children num_seniors : ℕ) 
  (flat_deposit adult_charge child_charge senior_charge service_charge : ℕ) 
  (split_bill : Bool) : ℕ :=
  flat_deposit + 
  num_adults * adult_charge + 
  num_children * child_charge + 
  num_seniors * senior_charge +
  (if split_bill then service_charge else 0)

/-- Theorem: The deposit for Hans' reservation is $101 --/
theorem hans_deposit_is_101 : 
  calculate_deposit 10 2 3 25 5 2 4 10 true = 101 := by
  sorry

end NUMINAMATH_CALUDE_hans_deposit_is_101_l2905_290542


namespace NUMINAMATH_CALUDE_distribute_10_8_l2905_290500

/-- The number of ways to distribute n indistinguishable objects into k distinguishable bins,
    with each bin receiving at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem stating that distributing 10 objects into 8 bins, with each bin receiving at least one,
    results in 36 possible distributions. -/
theorem distribute_10_8 : distribute 10 8 = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_10_8_l2905_290500


namespace NUMINAMATH_CALUDE_max_value_theorem_l2905_290587

theorem max_value_theorem (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 ≠ 0) :
  (|a + 2*b + 3*c| / Real.sqrt (a^2 + b^2 + c^2)) ≤ Real.sqrt 2 ∧
  ∃ (a' b' c' : ℝ), a' + b' + c' = 0 ∧ a'^2 + b'^2 + c'^2 ≠ 0 ∧
    |a' + 2*b' + 3*c'| / Real.sqrt (a'^2 + b'^2 + c'^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2905_290587


namespace NUMINAMATH_CALUDE_complex_division_simplification_l2905_290518

theorem complex_division_simplification : 
  let i : ℂ := Complex.I
  (2 * i) / (1 + i) = 1 + i := by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l2905_290518


namespace NUMINAMATH_CALUDE_joke_spread_after_one_minute_l2905_290564

def joke_spread (base : ℕ) (intervals : ℕ) : ℕ :=
  (base^(intervals + 1) - 1) / (base - 1)

theorem joke_spread_after_one_minute :
  joke_spread 6 6 = 55987 :=
by sorry

end NUMINAMATH_CALUDE_joke_spread_after_one_minute_l2905_290564


namespace NUMINAMATH_CALUDE_number_problem_l2905_290596

theorem number_problem (A B : ℝ) (h1 : A - B = 144) (h2 : A = 3 * B - 14) : A = 223 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2905_290596


namespace NUMINAMATH_CALUDE_indeterminate_b_value_l2905_290511

theorem indeterminate_b_value (a b c d : ℝ) : 
  a > b ∧ b > c ∧ c > d → 
  (a + b + c + d) / 4 = 12.345 → 
  ¬(∀ x : ℝ, x = b → (x > 12.345 ∨ x < 12.345 ∨ x = 12.345)) :=
by sorry

end NUMINAMATH_CALUDE_indeterminate_b_value_l2905_290511


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l2905_290524

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x * y = 4) (h2 : x - y = 2) : x^4 + y^4 = 112 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l2905_290524


namespace NUMINAMATH_CALUDE_right_triangle_angle_calculation_l2905_290561

theorem right_triangle_angle_calculation (x : ℝ) : 
  (3 * x > 3 * x - 40) →  -- Smallest angle condition
  (3 * x + (3 * x - 40) + 90 = 180) →  -- Sum of angles in a triangle
  x = 65 / 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_angle_calculation_l2905_290561


namespace NUMINAMATH_CALUDE_right_triangle_sides_l2905_290528

theorem right_triangle_sides : ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  a = 5 → b = 12 → c = 13 →
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l2905_290528


namespace NUMINAMATH_CALUDE_combination_equality_l2905_290579

theorem combination_equality (x : ℕ) : 
  (Nat.choose 7 x = Nat.choose 6 5 + Nat.choose 6 4) → (x = 2 ∨ x = 5) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l2905_290579


namespace NUMINAMATH_CALUDE_smallest_n_with_properties_l2905_290527

theorem smallest_n_with_properties : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 2 * n = k^2) ∧ 
  (∃ (l : ℕ), 3 * n = l^3) ∧ 
  (∃ (m : ℕ), 5 * n = m^5) ∧ 
  (∀ (x : ℕ), x > 0 ∧ 
    (∃ (k : ℕ), 2 * x = k^2) ∧ 
    (∃ (l : ℕ), 3 * x = l^3) ∧ 
    (∃ (m : ℕ), 5 * x = m^5) → 
    x ≥ n) ∧ 
  n = 11250 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_properties_l2905_290527


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l2905_290594

/-- A function that checks if a quadratic equation with given coefficients has rational solutions -/
def has_rational_solutions (a b c : ℤ) : Prop :=
  ∃ x : ℚ, a * x^2 + b * x + c = 0

/-- The set of positive integers k for which 3x^2 + 17x + k = 0 has rational solutions -/
def valid_k_set : Set ℤ :=
  {k : ℤ | k > 0 ∧ has_rational_solutions 3 17 k}

theorem quadratic_rational_solutions :
  ∃ k₁ k₂ : ℕ,
    k₁ ≠ k₂ ∧
    (↑k₁ : ℤ) ∈ valid_k_set ∧
    (↑k₂ : ℤ) ∈ valid_k_set ∧
    valid_k_set = {↑k₁, ↑k₂} ∧
    k₁ * k₂ = 240 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l2905_290594


namespace NUMINAMATH_CALUDE_exchange_calculation_l2905_290550

/-- Exchange rate between lire and dollars -/
def exchange_rate : ℚ := 2500 / 2

/-- Amount of dollars to be exchanged -/
def dollars_to_exchange : ℚ := 5

/-- Function to calculate lire received for a given amount of dollars -/
def lire_received (dollars : ℚ) : ℚ := dollars * exchange_rate

theorem exchange_calculation :
  lire_received dollars_to_exchange = 6250 := by
  sorry

end NUMINAMATH_CALUDE_exchange_calculation_l2905_290550


namespace NUMINAMATH_CALUDE_cubic_inequality_and_fraction_bound_l2905_290566

theorem cubic_inequality_and_fraction_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^3 + y^3 ≥ x^2*y + y^2*x) ∧
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a/b^2 + b/a^2 ≥ m/2 * (1/a + 1/b)) → m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_and_fraction_bound_l2905_290566


namespace NUMINAMATH_CALUDE_sqrt_74_between_consecutive_integers_product_l2905_290547

theorem sqrt_74_between_consecutive_integers_product : ∃ (n : ℕ), 
  n > 0 ∧ 
  (n : ℝ) < Real.sqrt 74 ∧ 
  Real.sqrt 74 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 72 := by
sorry

end NUMINAMATH_CALUDE_sqrt_74_between_consecutive_integers_product_l2905_290547


namespace NUMINAMATH_CALUDE_two_sector_area_l2905_290597

theorem two_sector_area (r : ℝ) (h : r = 15) : 
  2 * (45 / 360) * (π * r^2) = 56.25 * π := by
  sorry

end NUMINAMATH_CALUDE_two_sector_area_l2905_290597


namespace NUMINAMATH_CALUDE_original_profit_margin_l2905_290583

theorem original_profit_margin
  (original_price : ℝ)
  (original_margin : ℝ)
  (h_price_decrease : ℝ → ℝ → Prop)
  (h_margin_increase : ℝ → ℝ → Prop) :
  h_price_decrease original_price (original_price * (1 - 0.064)) →
  h_margin_increase original_margin (original_margin + 0.08) →
  original_margin = 0.17 :=
by sorry

end NUMINAMATH_CALUDE_original_profit_margin_l2905_290583


namespace NUMINAMATH_CALUDE_gcd_lcm_equality_implies_equal_l2905_290517

theorem gcd_lcm_equality_implies_equal (a b c : ℕ+) :
  (Nat.gcd a b + Nat.lcm a b = Nat.gcd a c + Nat.lcm a c) → b = c := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_equality_implies_equal_l2905_290517


namespace NUMINAMATH_CALUDE_dima_speed_ratio_l2905_290577

/-- Represents the time it takes Dima to walk from home to school -/
def walk_time : ℝ := 24

/-- Represents the time it takes Dima to run from home to school -/
def run_time : ℝ := 12

/-- Represents the time remaining before the school bell rings when Dima realizes he forgot his phone -/
def time_remaining : ℝ := 15

/-- States that Dima walks halfway to school before realizing he forgot his phone -/
axiom halfway_condition : walk_time / 2 = time_remaining - 3

/-- States that if Dima runs back home and then to school, he'll be 3 minutes late -/
axiom run_condition : run_time / 2 + run_time = time_remaining + 3

/-- States that if Dima runs back home and then walks to school, he'll be 15 minutes late -/
axiom run_walk_condition : run_time / 2 + walk_time = time_remaining + 15

/-- Theorem stating that Dima's running speed is twice his walking speed -/
theorem dima_speed_ratio : walk_time / run_time = 2 := by sorry

end NUMINAMATH_CALUDE_dima_speed_ratio_l2905_290577


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_is_one_l2905_290503

/-- Given three unit squares arranged in a straight line, each sharing a side with the next,
    where A is the bottom left vertex of the first square,
    B is the top right vertex of the second square,
    and C is the top left vertex of the third square,
    prove that the area of triangle ABC is 1. -/
theorem area_of_triangle_ABC_is_one :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 1)
  let C : ℝ × ℝ := (2, 1)
  let triangle_area (p q r : ℝ × ℝ) : ℝ :=
    (1/2) * abs (p.1 * (q.2 - r.2) + q.1 * (r.2 - p.2) + r.1 * (p.2 - q.2))
  triangle_area A B C = 1 := by
sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_is_one_l2905_290503


namespace NUMINAMATH_CALUDE_xy_xz_yz_bounds_l2905_290537

open Real

theorem xy_xz_yz_bounds (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  (∃ N n : ℝ, (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 → a * b + a * c + b * c ≤ N) ∧
              (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 → n ≤ a * b + a * c + b * c) ∧
              N = 75 ∧ n = 0) := by
  sorry

#check xy_xz_yz_bounds

end NUMINAMATH_CALUDE_xy_xz_yz_bounds_l2905_290537


namespace NUMINAMATH_CALUDE_running_reduction_is_five_l2905_290576

/-- Carly's running distances over four weeks -/
def running_distances : Fin 4 → ℚ
  | 0 => 2                        -- Week 1: 2 miles
  | 1 => 2 * 2 + 3                -- Week 2: twice as long as week 1 plus 3 extra miles
  | 2 => (2 * 2 + 3) * (9/7)      -- Week 3: 9/7 as much as week 2
  | 3 => 4                        -- Week 4: 4 miles due to injury

/-- The reduction in Carly's running distance when she was injured -/
def running_reduction : ℚ :=
  running_distances 2 - running_distances 3

theorem running_reduction_is_five :
  running_reduction = 5 := by sorry

end NUMINAMATH_CALUDE_running_reduction_is_five_l2905_290576


namespace NUMINAMATH_CALUDE_probability_of_specific_pair_l2905_290514

def total_items : ℕ := 4
def items_to_select : ℕ := 2
def favorable_outcomes : ℕ := 1

theorem probability_of_specific_pair :
  (favorable_outcomes : ℚ) / (total_items.choose items_to_select) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_pair_l2905_290514


namespace NUMINAMATH_CALUDE_vacation_cost_distribution_l2905_290563

/-- Represents the vacation cost distribution problem -/
theorem vacation_cost_distribution 
  (anna_paid ben_paid carol_paid dan_paid : ℚ)
  (a b c : ℚ)
  (h1 : anna_paid = 130)
  (h2 : ben_paid = 150)
  (h3 : carol_paid = 110)
  (h4 : dan_paid = 190)
  (h5 : (anna_paid + ben_paid + carol_paid + dan_paid) / 4 = 145)
  (h6 : a = 5)
  (h7 : b = 5)
  (h8 : c = 35)
  : a - b + c = 35 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_distribution_l2905_290563


namespace NUMINAMATH_CALUDE_burger_filler_percentage_l2905_290592

/-- Given a burger with specified total weight and filler weights, 
    calculate the percentage that is not filler -/
theorem burger_filler_percentage 
  (total_weight : ℝ) 
  (vegetable_filler : ℝ) 
  (grain_filler : ℝ) 
  (h1 : total_weight = 180) 
  (h2 : vegetable_filler = 45) 
  (h3 : grain_filler = 15) : 
  (total_weight - (vegetable_filler + grain_filler)) / total_weight = 2/3 := by
sorry

#eval (180 - (45 + 15)) / 180

end NUMINAMATH_CALUDE_burger_filler_percentage_l2905_290592


namespace NUMINAMATH_CALUDE_one_eighth_divided_by_one_fourth_l2905_290534

theorem one_eighth_divided_by_one_fourth (a b c : ℚ) :
  a = 1 / 8 → b = 1 / 4 → c = 1 / 2 → a / b = c := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_divided_by_one_fourth_l2905_290534


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2905_290519

theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  (a 2)^2 + 12*(a 2) - 8 = 0 →  -- a₂ is a root
  (a 10)^2 + 12*(a 10) - 8 = 0 →  -- a₁₀ is a root
  a 2 ≠ a 10 →  -- a₂ and a₁₀ are distinct roots
  a 6 = -6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2905_290519


namespace NUMINAMATH_CALUDE_base6_addition_problem_l2905_290530

-- Define a function to convert a base 6 number to base 10
def base6ToBase10 (d₂ d₁ d₀ : Nat) : Nat :=
  d₂ * 6^2 + d₁ * 6^1 + d₀ * 6^0

-- Define a function to convert a base 10 number to base 6
def base10ToBase6 (n : Nat) : Nat × Nat × Nat :=
  let d₂ := n / 36
  let r₂ := n % 36
  let d₁ := r₂ / 6
  let d₀ := r₂ % 6
  (d₂, d₁, d₀)

theorem base6_addition_problem :
  ∀ S H E : Nat,
    S ≠ 0 ∧ H ≠ 0 ∧ E ≠ 0 →
    S < 6 ∧ H < 6 ∧ E < 6 →
    S ≠ H ∧ S ≠ E ∧ H ≠ E →
    base6ToBase10 S H E + base6ToBase10 0 H E = base6ToBase10 E S H →
    S = 5 ∧ H = 4 ∧ E = 5 ∧ base10ToBase6 (S + H + E) = (2, 2, 0) := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_problem_l2905_290530


namespace NUMINAMATH_CALUDE_tan_sixty_degrees_l2905_290502

theorem tan_sixty_degrees : Real.tan (π / 3) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_sixty_degrees_l2905_290502


namespace NUMINAMATH_CALUDE_rosie_pies_from_36_apples_l2905_290509

/-- Given that Rosie can make three pies out of twelve apples, 
    this function calculates how many pies she can make from a given number of apples. -/
def pies_from_apples (apples : ℕ) : ℕ :=
  (apples * 3) / 12

theorem rosie_pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

#eval pies_from_apples 36

end NUMINAMATH_CALUDE_rosie_pies_from_36_apples_l2905_290509


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2905_290557

/-- An arithmetic sequence with first term 7, second term 11, and last term 95 has 23 terms. -/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ → ℕ), 
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
    a 0 = 7 →                                -- first term is 7
    a 1 = 11 →                               -- second term is 11
    (∃ m : ℕ, a m = 95 ∧ ∀ k > m, a k > 95) →  -- last term is 95
    ∃ n : ℕ, n = 23 ∧ a (n - 1) = 95 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2905_290557


namespace NUMINAMATH_CALUDE_sqrt_144000_simplification_l2905_290584

theorem sqrt_144000_simplification : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_144000_simplification_l2905_290584


namespace NUMINAMATH_CALUDE_inscribed_circles_radii_l2905_290555

/-- Three circles inscribed in a corner --/
structure InscribedCircles where
  r : ℝ  -- radius of the small circle
  a : ℝ  -- distance from center of small circle to corner vertex
  x : ℝ  -- radius of the medium circle
  y : ℝ  -- radius of the large circle

/-- The configuration of the inscribed circles --/
def valid_configuration (c : InscribedCircles) : Prop :=
  c.r > 0 ∧ c.a > c.r ∧ c.x > c.r ∧ c.y > c.x

/-- The theorem stating the radii of medium and large circles --/
theorem inscribed_circles_radii (c : InscribedCircles) 
  (h : valid_configuration c) : 
  c.x = (c.a * c.r) / (c.a - c.r) ∧ 
  c.y = (c.a^2 * c.r) / (c.a - c.r)^2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circles_radii_l2905_290555


namespace NUMINAMATH_CALUDE_point_on_axes_l2905_290571

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The coordinate axes in 2D space -/
def CoordinateAxes : Set Point2D :=
  {p : Point2D | p.x = 0 ∨ p.y = 0}

/-- Theorem: If xy = 0, then P(x,y) is located on the coordinate axes -/
theorem point_on_axes (p : Point2D) (h : p.x * p.y = 0) : p ∈ CoordinateAxes := by
  sorry

end NUMINAMATH_CALUDE_point_on_axes_l2905_290571


namespace NUMINAMATH_CALUDE_contrapositive_proof_l2905_290590

theorem contrapositive_proof (a b : ℝ) :
  (∀ a b, a > b → a - 5 > b - 5) ↔ (∀ a b, a - 5 ≤ b - 5 → a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_proof_l2905_290590


namespace NUMINAMATH_CALUDE_bread_cost_l2905_290560

/-- Proves that the cost of a loaf of bread is $2 given the specified conditions --/
theorem bread_cost (total_budget : ℝ) (candy_cost : ℝ) (turkey_proportion : ℝ) (money_left : ℝ)
  (h1 : total_budget = 32)
  (h2 : candy_cost = 2)
  (h3 : turkey_proportion = 1/3)
  (h4 : money_left = 18)
  : ∃ (bread_cost : ℝ),
    bread_cost = 2 ∧
    money_left = total_budget - candy_cost - turkey_proportion * (total_budget - candy_cost) - bread_cost :=
by
  sorry

end NUMINAMATH_CALUDE_bread_cost_l2905_290560


namespace NUMINAMATH_CALUDE_student_average_greater_than_true_average_l2905_290565

theorem student_average_greater_than_true_average
  (x y z w : ℝ) (h : x < y ∧ y < z ∧ z < w) :
  ((((x + y) / 2 + z) / 2) + w) / 2 > (x + y + z + w) / 4 :=
by sorry

end NUMINAMATH_CALUDE_student_average_greater_than_true_average_l2905_290565


namespace NUMINAMATH_CALUDE_negation_of_forall_x_squared_gt_one_l2905_290513

theorem negation_of_forall_x_squared_gt_one :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x₀ : ℝ, x₀^2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_x_squared_gt_one_l2905_290513


namespace NUMINAMATH_CALUDE_james_beverages_consumed_l2905_290556

/-- Represents the number of beverages James drinks in a week. -/
def beverages_consumed_in_week (
  soda_packs : ℕ)
  (sodas_per_pack : ℕ)
  (juice_packs : ℕ)
  (juices_per_pack : ℕ)
  (water_packs : ℕ)
  (waters_per_pack : ℕ)
  (energy_drinks : ℕ)
  (initial_sodas : ℕ)
  (initial_juices : ℕ)
  (mon_wed_sodas : ℕ)
  (mon_wed_juices : ℕ)
  (mon_wed_waters : ℕ)
  (thu_sun_sodas : ℕ)
  (thu_sun_juices : ℕ)
  (thu_sun_waters : ℕ)
  (thu_sun_energy : ℕ) : ℕ :=
  3 * (mon_wed_sodas + mon_wed_juices + mon_wed_waters) +
  4 * (thu_sun_sodas + thu_sun_juices + thu_sun_waters + thu_sun_energy)

/-- Proves that James drinks exactly 50 beverages in a week given the conditions. -/
theorem james_beverages_consumed :
  beverages_consumed_in_week 4 10 3 8 2 15 7 12 5 3 2 1 2 4 1 1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_james_beverages_consumed_l2905_290556


namespace NUMINAMATH_CALUDE_wendys_recycling_points_l2905_290546

/-- Wendy's recycling points calculation -/
theorem wendys_recycling_points :
  ∀ (points_per_bag : ℕ) (total_bags : ℕ) (unrecycled_bags : ℕ),
    points_per_bag = 5 →
    total_bags = 11 →
    unrecycled_bags = 2 →
    points_per_bag * (total_bags - unrecycled_bags) = 45 := by
  sorry

end NUMINAMATH_CALUDE_wendys_recycling_points_l2905_290546


namespace NUMINAMATH_CALUDE_sqrt_two_triangle_one_l2905_290568

-- Define the triangle operation
def triangle (a b : ℝ) : ℝ := a^2 - a*b

-- Theorem statement
theorem sqrt_two_triangle_one :
  triangle (Real.sqrt 2) 1 = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_triangle_one_l2905_290568


namespace NUMINAMATH_CALUDE_parabola_shift_l2905_290589

def original_function (x : ℝ) : ℝ := -2 * (x + 1)^2 + 5

def shift_left (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f (x + shift)

def shift_down (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f x - shift

def final_function (x : ℝ) : ℝ := -2 * (x + 3)^2 + 1

theorem parabola_shift :
  ∀ x : ℝ, shift_down (shift_left original_function 2) 4 x = final_function x :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l2905_290589


namespace NUMINAMATH_CALUDE_on_y_axis_on_x_axis_abscissa_greater_than_ordinate_l2905_290539

-- Define point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (2*m + 4, m - 1)

-- Theorem for condition (1)
theorem on_y_axis (m : ℝ) : 
  P m = (0, -3) ↔ P m = (0, (P m).2) :=
sorry

-- Theorem for condition (2)
theorem on_x_axis (m : ℝ) :
  P m = (6, 0) ↔ P m = ((P m).1, 0) :=
sorry

-- Theorem for condition (3)
theorem abscissa_greater_than_ordinate (m : ℝ) :
  P m = (-4, -5) ↔ (P m).1 = (P m).2 + 1 :=
sorry

end NUMINAMATH_CALUDE_on_y_axis_on_x_axis_abscissa_greater_than_ordinate_l2905_290539


namespace NUMINAMATH_CALUDE_min_good_operations_2009_l2905_290544

/-- Represents the sum of digits in the binary representation of a natural number -/
def S₂ (n : ℕ) : ℕ := sorry

/-- Represents the minimum number of "good" operations required to split a rope of length n into unit lengths -/
def min_good_operations (n : ℕ) : ℕ := sorry

/-- Theorem stating that the minimum number of good operations for a rope of length 2009 
    is equal to S₂(2009) - 1 -/
theorem min_good_operations_2009 : 
  min_good_operations 2009 = S₂ 2009 - 1 := by sorry

end NUMINAMATH_CALUDE_min_good_operations_2009_l2905_290544


namespace NUMINAMATH_CALUDE_age_equality_l2905_290569

/-- Proves that the number of years after which grandfather's age equals the sum of Xiaoming and father's ages is 14, given their current ages. -/
theorem age_equality (grandfather_age father_age xiaoming_age : ℕ) 
  (h1 : grandfather_age = 60)
  (h2 : father_age = 35)
  (h3 : xiaoming_age = 11) : 
  ∃ (years : ℕ), grandfather_age + years = (father_age + years) + (xiaoming_age + years) ∧ years = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_equality_l2905_290569


namespace NUMINAMATH_CALUDE_sequence_general_formula_l2905_290535

theorem sequence_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = 3^n - 2) →
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1)) →
  (∀ n : ℕ, n = 1 → a n = 1) ∧ 
  (∀ n : ℕ, n ≥ 2 → a n = 2 * 3^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_formula_l2905_290535


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l2905_290515

theorem multiplicative_inverse_modulo (A' B' : Nat) (m : Nat) (h : m = 2000000) :
  A' = 222222 →
  B' = 285714 →
  (1500000 * (A' * B')) % m = 1 :=
by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l2905_290515


namespace NUMINAMATH_CALUDE_smallest_addend_proof_l2905_290541

/-- The smallest non-negative integer that, when added to 27452, makes the sum divisible by 9 -/
def smallest_addend : ℕ := 7

/-- The original number we're working with -/
def original_number : ℕ := 27452

theorem smallest_addend_proof :
  (∀ k : ℕ, k < smallest_addend → ¬((original_number + k) % 9 = 0)) ∧
  ((original_number + smallest_addend) % 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_addend_proof_l2905_290541


namespace NUMINAMATH_CALUDE_sum_of_squares_l2905_290595

theorem sum_of_squares (x y z p q r : ℝ) 
  (h1 : x + y = p) 
  (h2 : y + z = q) 
  (h3 : z + x = r) 
  (hp : p ≠ 0) 
  (hq : q ≠ 0) 
  (hr : r ≠ 0) : 
  x^2 + y^2 + z^2 = (p^2 + q^2 + r^2 - p*q - q*r - r*p) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2905_290595


namespace NUMINAMATH_CALUDE_triangle_height_l2905_290598

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ)
  (h_area : area = 24)
  (h_base : base = 8)
  (h_triangle_area : area = (base * height) / 2) :
  height = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l2905_290598


namespace NUMINAMATH_CALUDE_sin_transformation_l2905_290531

theorem sin_transformation (x : ℝ) : 
  2 * Real.sin (x / 3 + π / 6) = 2 * Real.sin ((3 * x + π) / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_transformation_l2905_290531


namespace NUMINAMATH_CALUDE_right_triangle_semicircles_l2905_290506

theorem right_triangle_semicircles (P Q R : ℝ × ℝ) : 
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  (P.1 - Q.1) * (R.1 - Q.1) + (P.2 - Q.2) * (R.2 - Q.2) = 0 →  -- right angle at Q
  (1/2) * Real.pi * (pq/2)^2 = 50 * Real.pi →  -- area of semicircle on PQ
  Real.pi * (pr/2) = 18 * Real.pi →  -- circumference of semicircle on PR
  qr/2 = 20.6 ∧  -- radius of semicircle on QR
  ∃ (C : ℝ × ℝ), (C.1 - P.1)^2 + (C.2 - P.2)^2 = (pr/2)^2 ∧
                 (C.1 - R.1)^2 + (C.2 - R.2)^2 = (pr/2)^2 ∧
                 (C.1 - Q.1) * (R.1 - P.1) + (C.2 - Q.2) * (R.2 - P.2) = 0  -- 90° angle at Q in semicircle on PR
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_semicircles_l2905_290506


namespace NUMINAMATH_CALUDE_final_score_l2905_290510

def bullseye_points : ℕ := 50

def dart_throws (bullseye half_bullseye miss : ℕ) : Prop :=
  bullseye = bullseye_points ∧
  half_bullseye = bullseye_points / 2 ∧
  miss = 0

theorem final_score (bullseye half_bullseye miss : ℕ) 
  (h : dart_throws bullseye half_bullseye miss) : 
  bullseye + half_bullseye + miss = 75 := by
  sorry

end NUMINAMATH_CALUDE_final_score_l2905_290510


namespace NUMINAMATH_CALUDE_order_relation_l2905_290562

theorem order_relation (a b c : ℝ) : 
  a = 1 / 2023 ∧ 
  b = Real.tan (Real.exp (1 / 2023) / 2023) ∧ 
  c = Real.sin (Real.exp (1 / 2024) / 2024) →
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_order_relation_l2905_290562


namespace NUMINAMATH_CALUDE_proportion_equality_l2905_290551

-- Define variables a and b
variable (a b : ℝ)

-- Define the given condition
def condition : Prop := 2 * a = 5 * b

-- State the theorem to be proved
theorem proportion_equality (h : condition a b) : a / 5 = b / 2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l2905_290551


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l2905_290552

theorem prime_sum_theorem (p q : ℕ) : 
  Nat.Prime p → 
  Nat.Prime q → 
  Nat.Prime (7 * p + q) → 
  Nat.Prime (2 * q + 11) → 
  p^q + q^p = 17 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l2905_290552


namespace NUMINAMATH_CALUDE_intersection_M_N_l2905_290545

def M : Set ℝ := {1, 3, 4}
def N : Set ℝ := {x : ℝ | x^2 - 4*x + 3 = 0}

theorem intersection_M_N : M ∩ N = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2905_290545


namespace NUMINAMATH_CALUDE_problem_solution_l2905_290580

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sqrt 3 * Real.sin x + Real.cos x) - 1

theorem problem_solution :
  (∀ x ∈ Set.Icc 0 (π/4), f x ≤ 2) ∧
  (∀ x ∈ Set.Icc 0 (π/4), f x ≥ 1) ∧
  (∀ x₀ ∈ Set.Icc (π/4) (π/2), f x₀ = 6/5 → Real.cos (2*x₀) = (3 - 4*Real.sqrt 3)/10) ∧
  (∀ ω > 0, (∀ x ∈ Set.Ioo (π/3) (2*π/3), StrictMono (λ x => f (ω*x))) → ω ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2905_290580


namespace NUMINAMATH_CALUDE_library_visitors_l2905_290529

/-- Calculates the average number of visitors on non-Sunday days in a library -/
theorem library_visitors (total_days : Nat) (sunday_visitors : Nat) (avg_visitors : Nat) :
  total_days = 30 ∧ 
  sunday_visitors = 510 ∧ 
  avg_visitors = 285 →
  (total_days * avg_visitors - 5 * sunday_visitors) / 25 = 240 := by
sorry

end NUMINAMATH_CALUDE_library_visitors_l2905_290529


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2905_290521

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 12 - x^2 / 27 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = (2/3) * x ∨ y = -(2/3) * x

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y : ℝ, asymptote_equation x y ↔ (y^2 / x^2 = 4/9)) ∧
  hyperbola_equation 3 4 :=
sorry


end NUMINAMATH_CALUDE_hyperbola_properties_l2905_290521


namespace NUMINAMATH_CALUDE_fair_coin_probability_difference_l2905_290508

theorem fair_coin_probability_difference : 
  let n : ℕ := 5
  let p : ℚ := 1/2
  let prob_4_heads := (n.choose 4) * p^4 * (1-p)
  let prob_5_heads := p^n
  abs (prob_4_heads - prob_5_heads) = 9/32 := by
sorry

end NUMINAMATH_CALUDE_fair_coin_probability_difference_l2905_290508


namespace NUMINAMATH_CALUDE_no_geometric_triple_in_arithmetic_sequence_l2905_290586

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Define the property of containing 1 and √2
def contains_one_and_sqrt_two (a : ℕ → ℝ) : Prop :=
  ∃ k l : ℕ, a k = 1 ∧ a l = Real.sqrt 2

-- Define a geometric sequence of three terms
def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

-- Main theorem
theorem no_geometric_triple_in_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : contains_one_and_sqrt_two a) : 
  ¬∃ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ geometric_sequence (a i) (a j) (a k) :=
sorry

end NUMINAMATH_CALUDE_no_geometric_triple_in_arithmetic_sequence_l2905_290586


namespace NUMINAMATH_CALUDE_line_l_equation_l2905_290573

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ

-- Define the given conditions
def point_on_l : Point := (2, 3)
def L1 : Line := λ x y => 2*x - 5*y + 9
def L2 : Line := λ x y => 2*x - 5*y - 7
def midpoint_line : Line := λ x y => x - 4*y - 1

-- Define the line l
def l : Line := λ x y => 4*x - 5*y + 7

-- Theorem statement
theorem line_l_equation : 
  ∃ (A B : Point),
    (L1 A.1 A.2 = 0 ∧ L2 B.1 B.2 = 0) ∧ 
    (midpoint_line ((A.1 + B.1)/2) ((A.2 + B.2)/2) = 0) ∧
    (l point_on_l.1 point_on_l.2 = 0) ∧
    (∀ (x y : ℝ), l x y = 0 ↔ 4*x - 5*y + 7 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_l_equation_l2905_290573


namespace NUMINAMATH_CALUDE_next_simultaneous_occurrence_l2905_290523

def town_hall_interval : ℕ := 18
def fire_station_interval : ℕ := 24
def university_bell_interval : ℕ := 30

def simultaneous_occurrence (t : ℕ) : Prop :=
  t % town_hall_interval = 0 ∧
  t % fire_station_interval = 0 ∧
  t % university_bell_interval = 0

theorem next_simultaneous_occurrence :
  ∃ t : ℕ, t > 0 ∧ t ≤ 360 ∧ simultaneous_occurrence t ∧
  ∀ s : ℕ, 0 < s ∧ s < t → ¬simultaneous_occurrence s :=
sorry

end NUMINAMATH_CALUDE_next_simultaneous_occurrence_l2905_290523


namespace NUMINAMATH_CALUDE_grape_rate_proof_l2905_290591

theorem grape_rate_proof (grapes_kg mangoes_kg mangoes_rate total_paid : ℕ) 
  (h1 : grapes_kg = 8)
  (h2 : mangoes_kg = 9)
  (h3 : mangoes_rate = 65)
  (h4 : total_paid = 1145)
  : ∃ (grape_rate : ℕ), grape_rate * grapes_kg + mangoes_kg * mangoes_rate = total_paid ∧ grape_rate = 70 := by
  sorry

end NUMINAMATH_CALUDE_grape_rate_proof_l2905_290591


namespace NUMINAMATH_CALUDE_equation_solution_range_l2905_290567

theorem equation_solution_range (x m : ℝ) : 
  x / (x - 1) - 2 = (3 * m) / (2 * x - 2) → 
  x > 0 → 
  x ≠ 1 → 
  m < 4/3 ∧ m ≠ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2905_290567


namespace NUMINAMATH_CALUDE_even_product_probability_l2905_290581

-- Define the spinners
def spinner1 : List ℕ := [0, 2]
def spinner2 : List ℕ := [1, 3, 5]

-- Define a function to check if a number is even
def isEven (n : ℕ) : Bool := n % 2 = 0

-- Define a function to calculate the probability of an even product
def probEvenProduct (s1 s2 : List ℕ) : ℚ :=
  let totalOutcomes := s1.length * s2.length
  let evenOutcomes := (s1.filter isEven).length * s2.length
  evenOutcomes / totalOutcomes

-- Theorem statement
theorem even_product_probability :
  probEvenProduct spinner1 spinner2 = 1 := by sorry

end NUMINAMATH_CALUDE_even_product_probability_l2905_290581


namespace NUMINAMATH_CALUDE_right_triangle_area_in_circle_l2905_290549

/-- The area of a right triangle inscribed in a circle -/
theorem right_triangle_area_in_circle (r : ℝ) (h : r = 5) :
  let a : ℝ := 5 * (10 / 13)
  let b : ℝ := 12 * (10 / 13)
  let c : ℝ := 13 * (10 / 13)
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem
  (c = 2 * r) →        -- Diameter is the hypotenuse
  (1/2 * a * b = 6000/169) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_in_circle_l2905_290549


namespace NUMINAMATH_CALUDE_no_natural_squares_diff_2018_l2905_290585

theorem no_natural_squares_diff_2018 : ¬∃ (a b : ℕ), a^2 - b^2 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_squares_diff_2018_l2905_290585


namespace NUMINAMATH_CALUDE_line_l_equation_l2905_290516

-- Define the ellipse E
def ellipse (t : ℝ) (x y : ℝ) : Prop := x^2 / t + y^2 = 1

-- Define the parabola C
def parabola (x y : ℝ) : Prop := x^2 = 2 * Real.sqrt 2 * y

-- Define the point H
def H : ℝ × ℝ := (2, 0)

-- Define the condition for line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 2)

-- Define the condition for tangent lines being perpendicular
def perpendicular_tangents (x₁ x₂ : ℝ) : Prop := 
  (Real.sqrt 2 / 2 * x₁) * (Real.sqrt 2 / 2 * x₂) = -1

theorem line_l_equation :
  ∃ (t k : ℝ) (A B M N : ℝ × ℝ),
    -- Conditions
    (ellipse t A.1 A.2) ∧
    (ellipse t B.1 B.2) ∧
    (parabola A.1 A.2) ∧
    (parabola B.1 B.2) ∧
    (parabola M.1 M.2) ∧
    (parabola N.1 N.2) ∧
    (line_l k M.1 M.2) ∧
    (line_l k N.1 N.2) ∧
    (perpendicular_tangents M.1 N.1) →
    -- Conclusion
    k = -Real.sqrt 2 / 4 ∧ 
    ∀ (x y : ℝ), line_l k x y ↔ x + 2 * Real.sqrt 2 * y - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_l2905_290516


namespace NUMINAMATH_CALUDE_third_quadrant_trigonometric_sum_l2905_290540

theorem third_quadrant_trigonometric_sum (α : Real) : 
  (π < α ∧ α < 3*π/2) → 
  (|Real.sin (α/2)| / Real.sin (α/2)) + (|Real.cos (α/2)| / Real.cos (α/2)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_third_quadrant_trigonometric_sum_l2905_290540


namespace NUMINAMATH_CALUDE_simplify_expression_l2905_290526

theorem simplify_expression : 
  1 / (2 / (Real.sqrt 3 + 2) + 3 / (Real.sqrt 5 - 2)) = (10 + 2 * Real.sqrt 3 - 3 * Real.sqrt 5) / 43 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2905_290526


namespace NUMINAMATH_CALUDE_prob_standard_deck_l2905_290520

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)

/-- Calculates the probability of drawing either two queens or at least one king from a standard deck -/
def prob_two_queens_or_at_least_one_king (d : Deck) : ℚ :=
  sorry

/-- The main theorem stating the probability for a standard deck -/
theorem prob_standard_deck :
  let d : Deck := ⟨52, 4, 4⟩
  prob_two_queens_or_at_least_one_king d = 2/13 :=
sorry

end NUMINAMATH_CALUDE_prob_standard_deck_l2905_290520


namespace NUMINAMATH_CALUDE_reciprocal_expression_l2905_290593

theorem reciprocal_expression (m n : ℝ) (h : m * n = 1) : m * n^2 - (n - 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_expression_l2905_290593
