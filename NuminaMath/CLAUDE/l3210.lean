import Mathlib

namespace NUMINAMATH_CALUDE_moe_eating_time_l3210_321098

/-- The time taken for Moe to eat a certain number of cuttlebone pieces -/
theorem moe_eating_time (X : ℝ) : 
  (200 : ℝ) / 800 * X = X / 4 := by sorry

end NUMINAMATH_CALUDE_moe_eating_time_l3210_321098


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3210_321092

theorem quadratic_root_value (b : ℝ) : 
  (∀ x : ℝ, x^2 + Real.sqrt (b - 1) * x + b^2 - 4 = 0 → x = 0) →
  (b - 1 ≥ 0) →
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3210_321092


namespace NUMINAMATH_CALUDE_complex_multiplication_l3210_321000

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (3 + 2*i)*i = -2 + 3*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3210_321000


namespace NUMINAMATH_CALUDE_wood_frog_count_l3210_321007

theorem wood_frog_count (total : ℕ) (tree : ℕ) (poison : ℕ) (wood : ℕ) 
  (h1 : total = 78)
  (h2 : tree = 55)
  (h3 : poison = 10)
  (h4 : total = tree + poison + wood) : wood = 13 := by
  sorry

end NUMINAMATH_CALUDE_wood_frog_count_l3210_321007


namespace NUMINAMATH_CALUDE_square_equation_solution_l3210_321072

theorem square_equation_solution : 
  ∃ x : ℚ, ((3 * x + 15)^2 = 3 * (4 * x + 40)) ∧ (x = -5/3 ∨ x = -7) :=
by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3210_321072


namespace NUMINAMATH_CALUDE_workshop_analysis_l3210_321054

/-- Workshop worker information -/
structure Workshop where
  total_workers : ℕ
  avg_salary : ℝ
  technicians : ℕ
  technician_salary : ℝ
  managers : ℕ
  manager_salary : ℝ
  assistant_salary : ℝ

/-- Theorem about workshop workers and salaries -/
theorem workshop_analysis (w : Workshop)
  (h_total : w.total_workers = 20)
  (h_avg : w.avg_salary = 8000)
  (h_tech : w.technicians = 7)
  (h_tech_salary : w.technician_salary = 12000)
  (h_man : w.managers = 5)
  (h_man_salary : w.manager_salary = 15000)
  (h_assist_salary : w.assistant_salary = 6000) :
  let assistants := w.total_workers - w.technicians - w.managers
  let tech_man_total := w.technicians * w.technician_salary + w.managers * w.manager_salary
  let tech_man_avg := tech_man_total / (w.technicians + w.managers : ℝ)
  assistants = 8 ∧ tech_man_avg = 13250 := by
  sorry


end NUMINAMATH_CALUDE_workshop_analysis_l3210_321054


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_1_f_extrema_on_interval_2_l3210_321076

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Theorem for the interval [-2, 0]
theorem f_extrema_on_interval_1 :
  (∀ x ∈ Set.Icc (-2) 0, f x ≤ 2) ∧
  (∀ x ∈ Set.Icc (-2) 0, f x ≥ 10) ∧
  (∃ x ∈ Set.Icc (-2) 0, f x = 2) ∧
  (∃ x ∈ Set.Icc (-2) 0, f x = 10) :=
sorry

-- Theorem for the interval [2, 3]
theorem f_extrema_on_interval_2 :
  (∀ x ∈ Set.Icc 2 3, f x ≤ 5) ∧
  (∀ x ∈ Set.Icc 2 3, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 2 3, f x = 5) ∧
  (∃ x ∈ Set.Icc 2 3, f x = 2) :=
sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_1_f_extrema_on_interval_2_l3210_321076


namespace NUMINAMATH_CALUDE_smallest_factor_for_cube_l3210_321042

theorem smallest_factor_for_cube (n : ℕ) : n > 0 ∧ n * 49 = (7 : ℕ)^3 ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬∃ k : ℕ, m * 49 = k^3 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_for_cube_l3210_321042


namespace NUMINAMATH_CALUDE_point_A_coordinates_l3210_321002

-- Define the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Define the translation operations
def translateLeft (p : Point) (d : ℝ) : Point :=
  { x := p.x - d, y := p.y }

def translateUp (p : Point) (d : ℝ) : Point :=
  { x := p.x, y := p.y + d }

-- Theorem statement
theorem point_A_coordinates 
  (A : Point) 
  (B : Point)
  (C : Point)
  (hB : ∃ d : ℝ, translateLeft A d = B)
  (hC : ∃ d : ℝ, translateUp A d = C)
  (hBcoord : B.x = 1 ∧ B.y = 2)
  (hCcoord : C.x = 3 ∧ C.y = 4) :
  A.x = 3 ∧ A.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l3210_321002


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l3210_321079

theorem quadratic_form_ratio (j : ℝ) : 
  ∃ (c p q : ℝ), 8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q ∧ q / p = -151 / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l3210_321079


namespace NUMINAMATH_CALUDE_cs_candidates_count_l3210_321028

theorem cs_candidates_count (m : ℕ) (n : ℕ) : 
  m = 4 → 
  m * (n.choose 2) = 84 → 
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_cs_candidates_count_l3210_321028


namespace NUMINAMATH_CALUDE_exists_n_plus_Sn_eq_1980_consecutive_n_plus_Sn_l3210_321017

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Theorem 1: There exists a natural number n such that n + S(n) = 1980
theorem exists_n_plus_Sn_eq_1980 : ∃ n : ℕ, n + S n = 1980 := by sorry

-- Theorem 2: For any natural number k, either k or k+1 can be expressed as n + S(n)
theorem consecutive_n_plus_Sn : ∀ k : ℕ, (∃ n : ℕ, n + S n = k) ∨ (∃ n : ℕ, n + S n = k + 1) := by sorry

end NUMINAMATH_CALUDE_exists_n_plus_Sn_eq_1980_consecutive_n_plus_Sn_l3210_321017


namespace NUMINAMATH_CALUDE_max_product_sum_300_l3210_321024

theorem max_product_sum_300 : 
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 ∧ ∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l3210_321024


namespace NUMINAMATH_CALUDE_sandras_sweets_l3210_321066

theorem sandras_sweets (saved : ℚ) (mother_gave : ℚ) (father_gave : ℚ)
  (candy_cost : ℚ) (jelly_bean_cost : ℚ) (candy_count : ℕ) (jelly_bean_count : ℕ)
  (remaining : ℚ) :
  saved = 10 →
  mother_gave = 4 →
  candy_cost = 1/2 →
  jelly_bean_cost = 1/5 →
  candy_count = 14 →
  jelly_bean_count = 20 →
  remaining = 11 →
  saved + mother_gave + father_gave = 
    candy_cost * candy_count + jelly_bean_cost * jelly_bean_count + remaining →
  father_gave / mother_gave = 2 := by
sorry

#eval (8 : ℚ) / (4 : ℚ) -- Expected output: 2

end NUMINAMATH_CALUDE_sandras_sweets_l3210_321066


namespace NUMINAMATH_CALUDE_divisor_count_equality_implies_even_l3210_321020

/-- The number of positive integer divisors of n -/
def s (n : ℕ+) : ℕ := sorry

/-- If there exist positive integers a, b, and k such that k = s(a) = s(b) = s(2a+3b), then k must be even -/
theorem divisor_count_equality_implies_even (a b k : ℕ+) :
  k = s a ∧ k = s b ∧ k = s (2 * a + 3 * b) → Even k := by sorry

end NUMINAMATH_CALUDE_divisor_count_equality_implies_even_l3210_321020


namespace NUMINAMATH_CALUDE_call_duration_l3210_321019

def calls_per_year : ℕ := 52
def cost_per_minute : ℚ := 5 / 100
def total_cost_per_year : ℚ := 78

theorem call_duration :
  (total_cost_per_year / cost_per_minute) / calls_per_year = 30 := by
  sorry

end NUMINAMATH_CALUDE_call_duration_l3210_321019


namespace NUMINAMATH_CALUDE_negation_of_forall_inequality_negation_of_specific_inequality_l3210_321075

theorem negation_of_forall_inequality (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, x ≤ 1 → p x) ↔ (∃ x : ℝ, x ≤ 1 ∧ ¬(p x)) := by sorry

theorem negation_of_specific_inequality :
  (¬ ∀ x : ℝ, x ≤ 1 → x^2 - 2*x + 1 ≥ 0) ↔ (∃ x : ℝ, x ≤ 1 ∧ x^2 - 2*x + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_inequality_negation_of_specific_inequality_l3210_321075


namespace NUMINAMATH_CALUDE_paintings_on_last_page_paintings_on_last_page_zero_l3210_321043

theorem paintings_on_last_page (initial_albums : Nat) (pages_per_album : Nat) 
  (initial_paintings_per_page : Nat) (new_paintings_per_page : Nat) 
  (filled_albums : Nat) (filled_pages_last_album : Nat) : Nat :=
  let total_paintings := initial_albums * pages_per_album * initial_paintings_per_page
  let total_pages_filled := filled_albums * pages_per_album + filled_pages_last_album
  total_paintings - (total_pages_filled * new_paintings_per_page)

theorem paintings_on_last_page_zero : 
  paintings_on_last_page 10 36 8 9 6 28 = 0 := by
  sorry

end NUMINAMATH_CALUDE_paintings_on_last_page_paintings_on_last_page_zero_l3210_321043


namespace NUMINAMATH_CALUDE_goldfish_graph_is_finite_distinct_points_l3210_321011

/-- Represents the cost of purchasing goldfish -/
def goldfish_cost (n : ℕ) : ℚ :=
  if n ≥ 3 then 20 * n else 0

/-- The set of points representing goldfish purchases from 3 to 15 -/
def goldfish_graph : Set (ℕ × ℚ) :=
  {p | ∃ n : ℕ, 3 ≤ n ∧ n ≤ 15 ∧ p = (n, goldfish_cost n)}

theorem goldfish_graph_is_finite_distinct_points :
  Finite goldfish_graph ∧ ∀ p q : (ℕ × ℚ), p ∈ goldfish_graph → q ∈ goldfish_graph → p ≠ q → p.1 ≠ q.1 :=
sorry

end NUMINAMATH_CALUDE_goldfish_graph_is_finite_distinct_points_l3210_321011


namespace NUMINAMATH_CALUDE_evaluate_expression_l3210_321047

theorem evaluate_expression (a : ℝ) (h : a = 3) : (5 * a^2 - 11 * a + 6) * (2 * a - 4) = 36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3210_321047


namespace NUMINAMATH_CALUDE_neznaika_claim_incorrect_l3210_321082

def car_meeting_problem (s₁ s₂ : ℝ) : Prop :=
  s₁ > 0 ∧ s₂ > 0 ∧  -- Both speeds are positive
  s₁ + s₂ = 90 ∧     -- They meet after 1 hour
  2 * (s₁ + s₂) - 90 = 180  -- Total distance traveled in 2 hours

theorem neznaika_claim_incorrect :
  ∀ s₁ s₂ : ℝ, car_meeting_problem s₁ s₂ → s₁ + s₂ ≠ 60 :=
by sorry

end NUMINAMATH_CALUDE_neznaika_claim_incorrect_l3210_321082


namespace NUMINAMATH_CALUDE_digit_move_equals_multiply_divide_l3210_321087

def N : ℕ := 2173913043478260869565

theorem digit_move_equals_multiply_divide :
  (N * 4) / 5 = (N % 10^22) * 10 + (N / 10^22) :=
by sorry

end NUMINAMATH_CALUDE_digit_move_equals_multiply_divide_l3210_321087


namespace NUMINAMATH_CALUDE_system_solution_unique_l3210_321067

theorem system_solution_unique :
  ∃! (x y : ℝ), x + y = 15 ∧ x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3210_321067


namespace NUMINAMATH_CALUDE_xy_value_l3210_321012

theorem xy_value (x y : ℝ) (h : Real.sqrt (x + 2) + (y - Real.sqrt 3) ^ 2 = 0) : 
  x * y = -2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l3210_321012


namespace NUMINAMATH_CALUDE_root_implies_range_m_l3210_321045

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem root_implies_range_m :
  ∀ m : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ f x = m) → m ∈ Set.Icc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_root_implies_range_m_l3210_321045


namespace NUMINAMATH_CALUDE_complex_fraction_real_l3210_321031

theorem complex_fraction_real (m : ℝ) : 
  (((1 : ℂ) + m * Complex.I) / ((1 : ℂ) + Complex.I)).im = 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l3210_321031


namespace NUMINAMATH_CALUDE_tank_filling_time_l3210_321096

/-- Proves that the first pipe takes 5 hours to fill the tank alone given the conditions of the problem -/
theorem tank_filling_time (T : ℝ) 
  (h1 : T > 0)  -- Ensuring T is positive
  (h2 : 1/T + 1/4 - 1/20 = 1/2.5) : T = 5 := by
  sorry


end NUMINAMATH_CALUDE_tank_filling_time_l3210_321096


namespace NUMINAMATH_CALUDE_sum_transformation_l3210_321099

theorem sum_transformation (xs : List ℝ) 
  (h1 : xs.sum = 40)
  (h2 : (xs.map (λ x => 1 - x)).sum = 20) :
  (xs.map (λ x => 1 + x)).sum = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_transformation_l3210_321099


namespace NUMINAMATH_CALUDE_minimal_fraction_difference_l3210_321046

theorem minimal_fraction_difference (p q : ℕ+) : 
  (4 : ℚ) / 7 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < 7 / 12 ∧ 
  (∀ p' q' : ℕ+, (4 : ℚ) / 7 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < 7 / 12 → q ≤ q') →
  q - p = 8 := by
sorry

end NUMINAMATH_CALUDE_minimal_fraction_difference_l3210_321046


namespace NUMINAMATH_CALUDE_equation_solution_l3210_321053

theorem equation_solution : ∃ x : ℝ, 15 * 2 = x - 3 + 5 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3210_321053


namespace NUMINAMATH_CALUDE_square_difference_product_l3210_321049

theorem square_difference_product : (476 + 424)^2 - 4 * 476 * 424 = 4624 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_product_l3210_321049


namespace NUMINAMATH_CALUDE_friday_temperature_l3210_321074

def monday_temp : ℝ := 40

theorem friday_temperature 
  (h1 : (monday_temp + tuesday_temp + wednesday_temp + thursday_temp) / 4 = 48)
  (h2 : (tuesday_temp + wednesday_temp + thursday_temp + friday_temp) / 4 = 46) :
  friday_temp = 32 := by
sorry

end NUMINAMATH_CALUDE_friday_temperature_l3210_321074


namespace NUMINAMATH_CALUDE_vacation_pictures_remaining_l3210_321008

-- Define the number of pictures taken at each location
def zoo_pictures : ℕ := 49
def museum_pictures : ℕ := 8

-- Define the number of deleted pictures
def deleted_pictures : ℕ := 38

-- Theorem to prove
theorem vacation_pictures_remaining :
  zoo_pictures + museum_pictures - deleted_pictures = 19 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_remaining_l3210_321008


namespace NUMINAMATH_CALUDE_smallest_integer_with_consecutive_sums_l3210_321006

theorem smallest_integer_with_consecutive_sums : ∃ n : ℕ, 
  (∃ a : ℤ, n = (9 * a + 36)) ∧ 
  (∃ b : ℤ, n = (10 * b + 45)) ∧ 
  (∃ c : ℤ, n = (11 * c + 55)) ∧ 
  (∀ m : ℕ, m < n → 
    (¬∃ x : ℤ, m = (9 * x + 36)) ∨ 
    (¬∃ y : ℤ, m = (10 * y + 45)) ∨ 
    (¬∃ z : ℤ, m = (11 * z + 55))) ∧ 
  n = 495 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_consecutive_sums_l3210_321006


namespace NUMINAMATH_CALUDE_combine_like_terms_l3210_321015

theorem combine_like_terms (a b : ℝ) : 
  2 * a^3 * b - (1/2) * a^3 * b - a^2 * b + (1/2) * a^2 * b - a * b^2 = 
  (3/2) * a^3 * b - (1/2) * a^2 * b - a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l3210_321015


namespace NUMINAMATH_CALUDE_congruence_problem_l3210_321093

theorem congruence_problem (x : ℤ) : 
  (4 * x + 9) % 25 = 3 → (3 * x + 14) % 25 = 22 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3210_321093


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_three_minus_five_l3210_321059

theorem sqrt_two_times_sqrt_three_minus_five (x : ℝ) :
  x = Real.sqrt 2 * Real.sqrt 3 - 5 → x = Real.sqrt 6 - 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_three_minus_five_l3210_321059


namespace NUMINAMATH_CALUDE_caravan_camels_l3210_321044

theorem caravan_camels (hens goats keepers : ℕ) (camel_feet : ℕ) : 
  hens = 50 → 
  goats = 45 → 
  keepers = 15 → 
  camel_feet = (hens + goats + keepers + 224) * 2 - (hens * 2 + goats * 4 + keepers * 2) → 
  camel_feet / 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_caravan_camels_l3210_321044


namespace NUMINAMATH_CALUDE_omega_squared_plus_omega_plus_one_eq_zero_l3210_321068

theorem omega_squared_plus_omega_plus_one_eq_zero :
  let ω : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I
  ω^2 + ω + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_omega_squared_plus_omega_plus_one_eq_zero_l3210_321068


namespace NUMINAMATH_CALUDE_girls_not_adjacent_arrangements_l3210_321025

def num_boys : ℕ := 3
def num_girls : ℕ := 2

theorem girls_not_adjacent_arrangements :
  (num_boys.factorial * (num_boys + 1).choose num_girls) = 72 :=
by sorry

end NUMINAMATH_CALUDE_girls_not_adjacent_arrangements_l3210_321025


namespace NUMINAMATH_CALUDE_triangle_cinema_seats_l3210_321040

/-- Represents a triangular seating arrangement in a cinema --/
structure TriangularCinema where
  best_seat_number : ℕ
  total_seats : ℕ

/-- Checks if a given TriangularCinema configuration is valid --/
def is_valid_cinema (c : TriangularCinema) : Prop :=
  ∃ n : ℕ,
    -- The number of rows is 2n + 1
    (2 * n + 1) * ((2 * n + 1) + 1) / 2 = c.total_seats ∧
    -- The best seat is in the middle row
    (n + 1) * (n + 2) / 2 = c.best_seat_number

/-- Theorem stating the relationship between the best seat number and total seats --/
theorem triangle_cinema_seats (c : TriangularCinema) :
  c.best_seat_number = 265 → is_valid_cinema c → c.total_seats = 1035 := by
  sorry

#check triangle_cinema_seats

end NUMINAMATH_CALUDE_triangle_cinema_seats_l3210_321040


namespace NUMINAMATH_CALUDE_quadratic_root_shift_l3210_321036

theorem quadratic_root_shift (p q : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) ∧ (x₂^2 + p*x₂ + q = 0) →
  ((x₁ + 1)^2 + (p - 2)*(x₁ + 1) + (q - p + 1) = 0) ∧
  ((x₂ + 1)^2 + (p - 2)*(x₂ + 1) + (q - p + 1) = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_shift_l3210_321036


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_fractional_part_smallest_cube_root_exists_smallest_cube_root_is_68922_l3210_321021

theorem smallest_cube_root_with_fractional_part (m : ℕ) : 
  (∃ (n : ℕ) (r : ℝ), 
    n > 0 ∧ 
    r > 0 ∧ 
    r < 1/5000 ∧ 
    m^(1/3 : ℝ) = n + r) →
  m ≥ 68922 :=
by sorry

theorem smallest_cube_root_exists : 
  ∃ (n : ℕ) (r : ℝ), 
    n > 0 ∧ 
    r > 0 ∧ 
    r < 1/5000 ∧ 
    68922^(1/3 : ℝ) = n + r :=
by sorry

theorem smallest_cube_root_is_68922 : 
  (∀ m : ℕ, 
    (∃ (n : ℕ) (r : ℝ), 
      n > 0 ∧ 
      r > 0 ∧ 
      r < 1/5000 ∧ 
      m^(1/3 : ℝ) = n + r) →
    m ≥ 68922) ∧
  (∃ (n : ℕ) (r : ℝ), 
    n > 0 ∧ 
    r > 0 ∧ 
    r < 1/5000 ∧ 
    68922^(1/3 : ℝ) = n + r) :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_fractional_part_smallest_cube_root_exists_smallest_cube_root_is_68922_l3210_321021


namespace NUMINAMATH_CALUDE_ping_pong_dominating_subset_l3210_321050

/-- Represents a ping-pong match result between two players -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a team of ping-pong players -/
def Team := Fin 1000

/-- Represents the result of all matches between two teams -/
def MatchResults := Team → Team → MatchResult

theorem ping_pong_dominating_subset (results : MatchResults) :
  ∃ (dominating_team : Bool) (subset : Finset Team),
    subset.card ≤ 10 ∧
    ∀ (opponent : Team),
      ∃ (player : Team),
        player ∈ subset ∧
        ((dominating_team = true  ∧ results player opponent = MatchResult.Win) ∨
         (dominating_team = false ∧ results opponent player = MatchResult.Loss)) :=
sorry

end NUMINAMATH_CALUDE_ping_pong_dominating_subset_l3210_321050


namespace NUMINAMATH_CALUDE_average_points_is_27_l3210_321094

/-- Represents a hockey team's record --/
structure TeamRecord where
  wins : ℕ
  ties : ℕ

/-- Calculates the points for a team given their record --/
def calculatePoints (record : TeamRecord) : ℕ :=
  2 * record.wins + record.ties

/-- The number of teams in the playoffs --/
def numTeams : ℕ := 3

/-- The records of the three playoff teams --/
def team1 : TeamRecord := ⟨12, 4⟩
def team2 : TeamRecord := ⟨13, 1⟩
def team3 : TeamRecord := ⟨8, 10⟩

/-- Theorem: The average number of points for the playoff teams is 27 --/
theorem average_points_is_27 : 
  (calculatePoints team1 + calculatePoints team2 + calculatePoints team3) / numTeams = 27 := by
  sorry


end NUMINAMATH_CALUDE_average_points_is_27_l3210_321094


namespace NUMINAMATH_CALUDE_gauss_polynomial_reciprocal_l3210_321039

/-- Definition of a Gauss polynomial -/
def is_gauss_polynomial (g : ℤ → ℤ → (ℝ → ℝ)) : Prop :=
  ∀ (k l : ℤ) (x : ℝ), x ≠ 0 → x^(k*l) * g k l (1/x) = g k l x

/-- Theorem: Gauss polynomials are reciprocal -/
theorem gauss_polynomial_reciprocal (g : ℤ → ℤ → (ℝ → ℝ)) (h : is_gauss_polynomial g) :
  ∀ (k l : ℤ) (x : ℝ), x ≠ 0 → x^(k*l) * g k l (1/x) = g k l x :=
sorry

end NUMINAMATH_CALUDE_gauss_polynomial_reciprocal_l3210_321039


namespace NUMINAMATH_CALUDE_jane_mean_score_l3210_321005

def jane_scores : List ℝ := [98, 97, 92, 85, 93, 88, 82]

theorem jane_mean_score : 
  (jane_scores.sum / jane_scores.length : ℝ) = 90.71428571428571 := by
sorry

end NUMINAMATH_CALUDE_jane_mean_score_l3210_321005


namespace NUMINAMATH_CALUDE_decimal_to_binary_15_l3210_321051

theorem decimal_to_binary_15 : (15 : ℕ) = 0b1111 := by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_15_l3210_321051


namespace NUMINAMATH_CALUDE_total_cost_equation_l3210_321071

/-- Represents the total cost of tickets for a school trip to Green World -/
def totalCost (x : ℕ) : ℕ :=
  40 * x + 60

/-- Theorem stating the relationship between the number of students and the total cost -/
theorem total_cost_equation (x : ℕ) (y : ℕ) :
  y = totalCost x ↔ y = 40 * x + 60 := by sorry

end NUMINAMATH_CALUDE_total_cost_equation_l3210_321071


namespace NUMINAMATH_CALUDE_cost_of_type_B_books_cost_equals_formula_l3210_321001

/-- Given a total of 100 books to be purchased, with x books of type A,
    prove that the cost of purchasing type B books is 8(100-x) yuan,
    where the unit price of type B book is 8. -/
theorem cost_of_type_B_books (x : ℕ) : ℕ :=
  let total_books : ℕ := 100
  let unit_price_B : ℕ := 8
  let num_type_B : ℕ := total_books - x
  unit_price_B * num_type_B

#check cost_of_type_B_books

/-- Proof that the cost of type B books is 8(100-x) -/
theorem cost_equals_formula (x : ℕ) :
  cost_of_type_B_books x = 8 * (100 - x) :=
by sorry

#check cost_equals_formula

end NUMINAMATH_CALUDE_cost_of_type_B_books_cost_equals_formula_l3210_321001


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3210_321014

/-- The quadratic equation x^2 - 2x - 6 = 0 has two distinct real roots -/
theorem quadratic_two_distinct_roots : ∃ (x₁ x₂ : ℝ), 
  x₁ ≠ x₂ ∧ 
  x₁^2 - 2*x₁ - 6 = 0 ∧ 
  x₂^2 - 2*x₂ - 6 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3210_321014


namespace NUMINAMATH_CALUDE_not_cyclically_symmetric_example_cyclically_symmetric_example_difference_of_cyclically_symmetric_triangle_angles_cyclically_symmetric_l3210_321023

-- Definition of cyclically symmetric function
def CyclicallySymmetric (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c, f a b c = f b c a ∧ f a b c = f c a b

-- Statement 1
theorem not_cyclically_symmetric_example :
  ¬CyclicallySymmetric (fun x y z => x^2 - y^2 + z) := by sorry

-- Statement 2
theorem cyclically_symmetric_example :
  CyclicallySymmetric (fun x y z => x^2*(y-z) + y^2*(z-x) + z^2*(x-y)) := by sorry

-- Statement 3
theorem difference_of_cyclically_symmetric (f g : ℝ → ℝ → ℝ → ℝ) :
  CyclicallySymmetric f → CyclicallySymmetric g →
  CyclicallySymmetric (fun x y z => f x y z - g x y z) := by sorry

-- Statement 4
theorem triangle_angles_cyclically_symmetric (A B C : ℝ) :
  A + B + C = π →
  CyclicallySymmetric (fun x y z => 2 + Real.cos z * Real.cos (x-y) - Real.cos z^2) := by sorry

end NUMINAMATH_CALUDE_not_cyclically_symmetric_example_cyclically_symmetric_example_difference_of_cyclically_symmetric_triangle_angles_cyclically_symmetric_l3210_321023


namespace NUMINAMATH_CALUDE_difference_greater_than_twice_l3210_321090

theorem difference_greater_than_twice (a : ℝ) : 
  (∀ x, x - 5 > 2*x ↔ x = a) ↔ a - 5 > 2*a := by sorry

end NUMINAMATH_CALUDE_difference_greater_than_twice_l3210_321090


namespace NUMINAMATH_CALUDE_more_apples_than_pears_l3210_321060

theorem more_apples_than_pears :
  let total_fruits : ℕ := 85
  let num_apples : ℕ := 48
  let num_pears : ℕ := total_fruits - num_apples
  num_apples - num_pears = 11 :=
by sorry

end NUMINAMATH_CALUDE_more_apples_than_pears_l3210_321060


namespace NUMINAMATH_CALUDE_isabel_paper_problem_l3210_321057

theorem isabel_paper_problem (total : ℕ) (used : ℕ) (remaining : ℕ) : 
  total = 900 → used = 156 → remaining = total - used → remaining = 744 := by
  sorry

end NUMINAMATH_CALUDE_isabel_paper_problem_l3210_321057


namespace NUMINAMATH_CALUDE_complex_modulus_of_three_plus_i_squared_l3210_321027

theorem complex_modulus_of_three_plus_i_squared :
  let z : ℂ := (3 + Complex.I) ^ 2
  ‖z‖ = 10 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_of_three_plus_i_squared_l3210_321027


namespace NUMINAMATH_CALUDE_smallest_n_with_constant_term_l3210_321022

theorem smallest_n_with_constant_term : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → k < n → 
    ¬ ∃ (r : ℕ), r ≤ k ∧ 3 * k = (7 * r) / 2) ∧
  (∃ (r : ℕ), r ≤ n ∧ 3 * n = (7 * r) / 2) ∧
  n = 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_constant_term_l3210_321022


namespace NUMINAMATH_CALUDE_find_divisor_l3210_321034

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 199 → quotient = 11 → remainder = 1 →
  ∃ (divisor : Nat), dividend = divisor * quotient + remainder ∧ divisor = 18 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3210_321034


namespace NUMINAMATH_CALUDE_condition_A_right_triangle_condition_B_right_triangle_condition_C_not_right_triangle_condition_D_right_triangle_l3210_321081

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define functions to calculate side lengths and angles
def side_length (p q : ℝ × ℝ) : ℝ := sorry
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define a function to check if a triangle is right-angled
def is_right_triangle (t : Triangle) : Prop :=
  let a := side_length t.A t.B
  let b := side_length t.B t.C
  let c := side_length t.C t.A
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

-- Theorem for condition A
theorem condition_A_right_triangle (t : Triangle) :
  side_length t.A t.B = 3 ∧ side_length t.B t.C = 4 ∧ side_length t.C t.A = 5 →
  is_right_triangle t :=
sorry

-- Theorem for condition B
theorem condition_B_right_triangle (t : Triangle) (k : ℝ) :
  side_length t.A t.B = 3*k ∧ side_length t.B t.C = 4*k ∧ side_length t.C t.A = 5*k →
  is_right_triangle t :=
sorry

-- Theorem for condition C
theorem condition_C_not_right_triangle (t : Triangle) :
  ∃ (k : ℝ), angle t.B t.A t.C = 3*k ∧ angle t.C t.B t.A = 4*k ∧ angle t.A t.C t.B = 5*k →
  ¬ is_right_triangle t :=
sorry

-- Theorem for condition D
theorem condition_D_right_triangle (t : Triangle) :
  angle t.B t.A t.C = 40 ∧ angle t.C t.B t.A = 50 →
  is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_condition_A_right_triangle_condition_B_right_triangle_condition_C_not_right_triangle_condition_D_right_triangle_l3210_321081


namespace NUMINAMATH_CALUDE_applicants_age_standard_deviation_l3210_321037

/-- The standard deviation of applicants' ages given specific conditions -/
theorem applicants_age_standard_deviation 
  (average_age : ℝ)
  (max_different_ages : ℕ)
  (h_average : average_age = 30)
  (h_max_ages : max_different_ages = 15)
  (h_range : max_different_ages = 2 * standard_deviation)
  (standard_deviation : ℝ) :
  standard_deviation = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_applicants_age_standard_deviation_l3210_321037


namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l3210_321084

theorem quadratic_rational_root_even_coefficient 
  (a b c : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : ∃ (x : ℚ), a * x^2 + b * x + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l3210_321084


namespace NUMINAMATH_CALUDE_total_caps_production_l3210_321058

/-- The total number of caps produced over four weeks, given the production
    of the first three weeks and the fourth week being the average of the first three. -/
theorem total_caps_production
  (week1 : ℕ)
  (week2 : ℕ)
  (week3 : ℕ)
  (h1 : week1 = 320)
  (h2 : week2 = 400)
  (h3 : week3 = 300) :
  week1 + week2 + week3 + (week1 + week2 + week3) / 3 = 1360 := by
  sorry

#eval 320 + 400 + 300 + (320 + 400 + 300) / 3

end NUMINAMATH_CALUDE_total_caps_production_l3210_321058


namespace NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l3210_321078

theorem triangle_circumcircle_diameter 
  (a : Real) 
  (B : Real) 
  (S : Real) : 
  a = 1 → 
  B = π / 4 → 
  S = 2 → 
  ∃ (b c d : Real), 
    c = 4 * Real.sqrt 2 ∧ 
    b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ 
    b = 5 ∧ 
    d = b / (Real.sin B) ∧ 
    d = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l3210_321078


namespace NUMINAMATH_CALUDE_modulus_of_9_minus_40i_l3210_321009

theorem modulus_of_9_minus_40i : Complex.abs (9 - 40*I) = 41 := by sorry

end NUMINAMATH_CALUDE_modulus_of_9_minus_40i_l3210_321009


namespace NUMINAMATH_CALUDE_initial_number_proof_l3210_321088

theorem initial_number_proof (x : ℕ) : x - 109 = 109 + 68 → x = 286 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l3210_321088


namespace NUMINAMATH_CALUDE_probability_of_9_heads_in_12_flips_l3210_321041

def num_flips : ℕ := 12
def num_heads : ℕ := 9

theorem probability_of_9_heads_in_12_flips :
  (num_flips.choose num_heads : ℚ) / 2^num_flips = 55 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_9_heads_in_12_flips_l3210_321041


namespace NUMINAMATH_CALUDE_calculation_proof_l3210_321064

theorem calculation_proof : 
  Real.sqrt 27 - |2 * Real.sqrt 3 - 9 * Real.tan (30 * π / 180)| + (1/2)⁻¹ - (1 - π)^0 = 2 * Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3210_321064


namespace NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l3210_321013

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D :=
  {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xOy plane -/
def symmetricToXOyPlane (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = q.y ∧ p.z = -q.z

theorem symmetric_point_xoy_plane :
  let m : Point3D := ⟨2, 5, 8⟩
  let n : Point3D := ⟨2, 5, -8⟩
  symmetricToXOyPlane m n := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l3210_321013


namespace NUMINAMATH_CALUDE_shelf_filling_theorem_l3210_321063

theorem shelf_filling_theorem (A H S M E : ℕ) 
  (distinct : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ E ∧ 
              H ≠ S ∧ H ≠ M ∧ H ≠ E ∧ 
              S ≠ M ∧ S ≠ E ∧ 
              M ≠ E)
  (positive : A > 0 ∧ H > 0 ∧ S > 0 ∧ M > 0 ∧ E > 0)
  (thicker : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y > x ∧ 
             A * x + H * y = S * x + M * y ∧ 
             A * x + H * y = E * x) : 
  E = (A * M - S * H) / (M - H) :=
by sorry

end NUMINAMATH_CALUDE_shelf_filling_theorem_l3210_321063


namespace NUMINAMATH_CALUDE_sugar_trader_profit_l3210_321077

/-- Represents the profit calculation for a sugar trader. -/
theorem sugar_trader_profit (Q : ℝ) (C : ℝ) : Q > 0 → C > 0 → 
  (Q - 1200) * (1.08 * C) + 1200 * (1.12 * C) = Q * C * 1.11 → Q = 1600 := by
  sorry

#check sugar_trader_profit

end NUMINAMATH_CALUDE_sugar_trader_profit_l3210_321077


namespace NUMINAMATH_CALUDE_jerry_field_hours_eq_96_l3210_321010

/-- The number of hours Jerry spends at the field watching his daughters play and practice -/
def jerry_field_hours : ℕ :=
  let num_daughters : ℕ := 2
  let games_per_daughter : ℕ := 8
  let practice_hours_per_game : ℕ := 4
  let game_duration_hours : ℕ := 2
  
  let game_hours_per_daughter : ℕ := games_per_daughter * game_duration_hours
  let practice_hours_per_daughter : ℕ := games_per_daughter * practice_hours_per_game
  
  num_daughters * (game_hours_per_daughter + practice_hours_per_daughter)

theorem jerry_field_hours_eq_96 : jerry_field_hours = 96 := by
  sorry

end NUMINAMATH_CALUDE_jerry_field_hours_eq_96_l3210_321010


namespace NUMINAMATH_CALUDE_function_bounds_l3210_321016

/-- Given a function f(x) = 1 - a cos x - b sin x - A cos 2x - B sin 2x,
    where a, b, A, B are real constants, and f(x) ≥ 0 for all real x,
    prove that a² + b² ≤ 2 and A² + B² ≤ 1. -/
theorem function_bounds (a b A B : ℝ) 
    (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_bounds_l3210_321016


namespace NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_for_increasing_l3210_321048

-- Define a geometric sequence
def geometric_sequence (a₀ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n => a₀ * q^n

-- Define monotonically increasing sequence
def monotonically_increasing (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, s n ≤ s (n + 1)

-- Theorem statement
theorem q_gt_one_neither_sufficient_nor_necessary_for_increasing
  (a₀ : ℝ) (q : ℝ) :
  ¬(((q > 1) → monotonically_increasing (geometric_sequence a₀ q)) ∧
    (monotonically_increasing (geometric_sequence a₀ q) → (q > 1))) :=
by sorry

end NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_for_increasing_l3210_321048


namespace NUMINAMATH_CALUDE_motel_rental_rate_l3210_321004

theorem motel_rental_rate (lower_rate higher_rate total_rent : ℚ) 
  (h1 : lower_rate = 40)
  (h2 : total_rent = 400)
  (h3 : total_rent / 2 = total_rent - 10 * (higher_rate - lower_rate)) :
  higher_rate = 60 := by
  sorry

end NUMINAMATH_CALUDE_motel_rental_rate_l3210_321004


namespace NUMINAMATH_CALUDE_second_column_halving_matrix_l3210_321038

def halve_second_column (N : Matrix (Fin 2) (Fin 2) ℝ) (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∀ i j, (N * M) i j = if j = 1 then (1/2 : ℝ) * M i j else M i j

theorem second_column_halving_matrix :
  ∃ N : Matrix (Fin 2) (Fin 2) ℝ, 
    (N 0 0 = 1 ∧ N 0 1 = 0 ∧ N 1 0 = 0 ∧ N 1 1 = 1/2) ∧
    ∀ M : Matrix (Fin 2) (Fin 2) ℝ, halve_second_column N M :=
by
  sorry

end NUMINAMATH_CALUDE_second_column_halving_matrix_l3210_321038


namespace NUMINAMATH_CALUDE_fox_jeans_price_l3210_321080

/-- Regular price of Pony jeans in dollars -/
def pony_price : ℝ := 18

/-- Total savings on 5 pairs of jeans (3 Fox, 2 Pony) in dollars -/
def total_savings : ℝ := 8.91

/-- Sum of discount rates for Fox and Pony jeans as a percentage -/
def total_discount_rate : ℝ := 22

/-- Discount rate on Pony jeans as a percentage -/
def pony_discount_rate : ℝ := 10.999999999999996

/-- Regular price of Fox jeans in dollars -/
def fox_price : ℝ := 15

theorem fox_jeans_price : 
  ∃ (fox_discount_rate : ℝ),
    fox_discount_rate + pony_discount_rate = total_discount_rate ∧
    3 * (fox_price * fox_discount_rate / 100) + 
    2 * (pony_price * pony_discount_rate / 100) = total_savings :=
by sorry

end NUMINAMATH_CALUDE_fox_jeans_price_l3210_321080


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l3210_321018

/-- Represents the dimensions of a rectangular roof -/
structure RoofDimensions where
  width : ℝ
  length : ℝ
  area : ℝ
  length_width_ratio : length = 4 * width
  area_equation : area = length * width

/-- The difference between the length and width of the roof -/
def length_width_difference (roof : RoofDimensions) : ℝ :=
  roof.length - roof.width

/-- Theorem stating the approximate difference between length and width -/
theorem roof_dimension_difference : 
  ∃ (roof : RoofDimensions), 
    roof.area = 675 ∧ 
    (abs (length_width_difference roof - 38.97) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l3210_321018


namespace NUMINAMATH_CALUDE_next_coincidence_exact_next_coincidence_l3210_321089

def factory_whistle := 18
def train_bell := 24
def fire_alarm := 30

theorem next_coincidence (t : ℕ) : t > 0 ∧ t % factory_whistle = 0 ∧ t % train_bell = 0 ∧ t % fire_alarm = 0 → t ≥ 360 :=
sorry

theorem exact_next_coincidence : ∃ (t : ℕ), t = 360 ∧ t % factory_whistle = 0 ∧ t % train_bell = 0 ∧ t % fire_alarm = 0 :=
sorry

end NUMINAMATH_CALUDE_next_coincidence_exact_next_coincidence_l3210_321089


namespace NUMINAMATH_CALUDE_trigonometric_calculation_quadratic_equation_solution_l3210_321085

-- Problem 1
theorem trigonometric_calculation :
  3 * Real.tan (30 * π / 180) - Real.tan (45 * π / 180)^2 + 2 * Real.sin (60 * π / 180) = 2 * Real.sqrt 3 - 1 := by
  sorry

-- Problem 2
theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ (3*x - 1)*(x + 2) - (11*x - 4)
  (∃ x : ℝ, f x = 0) ↔ (f ((3 + Real.sqrt 3) / 3) = 0 ∧ f ((3 - Real.sqrt 3) / 3) = 0) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_calculation_quadratic_equation_solution_l3210_321085


namespace NUMINAMATH_CALUDE_max_value_theorem_l3210_321061

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 81/4 ∧
  ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧
    (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 81/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3210_321061


namespace NUMINAMATH_CALUDE_even_function_inequality_l3210_321003

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem even_function_inequality (h1 : IsEven f) (h2 : f 2 < f 3) : f (-3) > f (-2) := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3210_321003


namespace NUMINAMATH_CALUDE_parabola_symmetric_points_l3210_321065

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y = ax^2 - 1 -/
def parabola (a : ℝ) (p : Point) : Prop :=
  p.y = a * p.x^2 - 1

/-- Two points are symmetric with respect to the line x + y = 0 -/
def symmetricPoints (p q : Point) : Prop :=
  p.x + q.x + p.y + q.y = 0 ∧ p.x + p.y = -(q.x + q.y)

/-- Main theorem: There exist two distinct points on the parabola
    y = ax^2 - 1 that are symmetric with respect to x + y = 0
    if and only if a > 3/4 -/
theorem parabola_symmetric_points (a : ℝ) :
  (∃ p q : Point, p ≠ q ∧ parabola a p ∧ parabola a q ∧ symmetricPoints p q) ↔
  a > 3/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_symmetric_points_l3210_321065


namespace NUMINAMATH_CALUDE_largest_non_expressible_l3210_321026

def is_composite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_not_multiple_of_four (n : ℕ) : Prop :=
  ¬(∃ k, n = 4 * k)

def is_expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ is_composite b ∧ is_not_multiple_of_four b ∧ n = 36 * a + b

theorem largest_non_expressible : 
  (∀ n > 147, is_expressible n) ∧ ¬(is_expressible 147) :=
sorry

end NUMINAMATH_CALUDE_largest_non_expressible_l3210_321026


namespace NUMINAMATH_CALUDE_performance_arrangements_l3210_321097

/-- The number of performances of each type -/
def num_singing : ℕ := 2
def num_dance : ℕ := 3
def num_variety : ℕ := 3

/-- The total number of performances -/
def total_performances : ℕ := num_singing + num_dance + num_variety

/-- Number of ways to arrange performances with singing at beginning and end -/
def arrangement_singing_ends : ℕ := 1440

/-- Number of ways to arrange performances with non-adjacent singing -/
def arrangement_non_adjacent_singing : ℕ := 30240

/-- Number of ways to arrange performances with adjacent singing and non-adjacent dance -/
def arrangement_adjacent_singing_non_adjacent_dance : ℕ := 2880

theorem performance_arrangements :
  (total_performances = 8) →
  (arrangement_singing_ends = 1440) ∧
  (arrangement_non_adjacent_singing = 30240) ∧
  (arrangement_adjacent_singing_non_adjacent_dance = 2880) :=
by sorry

end NUMINAMATH_CALUDE_performance_arrangements_l3210_321097


namespace NUMINAMATH_CALUDE_increasing_on_open_interval_l3210_321055

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Assume f' is the derivative of f
variable (h : ∀ x, HasDerivAt f (f' x) x)

-- Theorem statement
theorem increasing_on_open_interval
  (h1 : ∀ x ∈ Set.Ioo 4 5, f' x > 0) :
  StrictMonoOn f (Set.Ioo 4 5) :=
sorry

end NUMINAMATH_CALUDE_increasing_on_open_interval_l3210_321055


namespace NUMINAMATH_CALUDE_largest_even_number_less_than_150_div_9_l3210_321030

theorem largest_even_number_less_than_150_div_9 :
  ∃ (x : ℕ), 
    x % 2 = 0 ∧ 
    9 * x < 150 ∧ 
    ∀ (y : ℕ), y % 2 = 0 → 9 * y < 150 → y ≤ x ∧
    x = 16 :=
by sorry

end NUMINAMATH_CALUDE_largest_even_number_less_than_150_div_9_l3210_321030


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3210_321091

def f (x : ℝ) : ℝ := x^2 + 1

theorem quadratic_function_properties :
  (f 0 = 1) ∧ (∀ x : ℝ, deriv f x > 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3210_321091


namespace NUMINAMATH_CALUDE_intersection_M_N_l3210_321029

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2}
def N : Set ℝ := {x | ∃ y, (x^2/2) + y^2 = 1}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = Set.Icc 0 (Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3210_321029


namespace NUMINAMATH_CALUDE_tree_distance_l3210_321032

/-- Given 8 equally spaced trees along a road, where the distance between
    the first and fifth tree is 100 feet, the distance between the first
    and last tree is 175 feet. -/
theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  let distance_between (i j : ℕ) := d * (j - i : ℝ) / 4
  distance_between 1 n = 175 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l3210_321032


namespace NUMINAMATH_CALUDE_sheep_distribution_l3210_321095

theorem sheep_distribution (A B C D : ℕ) : 
  C = D + 10 ∧ 
  (3 * C) / 4 + A = B + C / 4 + D ∧
  (∃ (x : ℕ), x > 0 ∧ 
    (2 * A) / 3 + (B + A / 3 - (B + A / 3) / 4) + 
    (C + (B + A / 3) / 4 - (C + (B + A / 3) / 4) / 5) + 
    (D + (C + (B + A / 3) / 4) / 5 + x) = 
    4 * ((2 * A) / 3 + (B + A / 3 - (B + A / 3) / 4) + x)) →
  A = 60 ∧ B = 50 ∧ C = 40 ∧ D = 30 := by
sorry

end NUMINAMATH_CALUDE_sheep_distribution_l3210_321095


namespace NUMINAMATH_CALUDE_parametric_to_standard_equation_l3210_321062

theorem parametric_to_standard_equation :
  ∀ (x y : ℝ), (∃ t : ℝ, x = 4 * t + 1 ∧ y = -2 * t - 5) → x + 2 * y + 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_standard_equation_l3210_321062


namespace NUMINAMATH_CALUDE_sequence_gcd_property_l3210_321035

theorem sequence_gcd_property (a : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) →
  ∀ i : ℕ, a i = i := by
  sorry

end NUMINAMATH_CALUDE_sequence_gcd_property_l3210_321035


namespace NUMINAMATH_CALUDE_base7_multiplication_l3210_321056

/-- Converts a base 7 number (represented as a list of digits) to a natural number. -/
def base7ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 7 * acc + d) 0

/-- Converts a natural number to its base 7 representation (as a list of digits). -/
def natToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- The main theorem stating that 356₇ * 4₇ = 21323₇ in base 7. -/
theorem base7_multiplication :
  natToBase7 (base7ToNat [3, 5, 6] * base7ToNat [4]) = [2, 1, 3, 2, 3] := by
  sorry

#eval base7ToNat [3, 5, 6] -- Should output 188
#eval base7ToNat [4] -- Should output 4
#eval natToBase7 (188 * 4) -- Should output [2, 1, 3, 2, 3]

end NUMINAMATH_CALUDE_base7_multiplication_l3210_321056


namespace NUMINAMATH_CALUDE_boys_exam_pass_count_l3210_321052

theorem boys_exam_pass_count :
  ∀ (total_boys : ℕ) 
    (avg_all avg_pass avg_fail : ℚ)
    (pass_count : ℕ),
  total_boys = 120 →
  avg_all = 35 →
  avg_pass = 39 →
  avg_fail = 15 →
  pass_count ≤ total_boys →
  (pass_count : ℚ) * avg_pass + (total_boys - pass_count : ℚ) * avg_fail = (total_boys : ℚ) * avg_all →
  pass_count = 100 := by
sorry

end NUMINAMATH_CALUDE_boys_exam_pass_count_l3210_321052


namespace NUMINAMATH_CALUDE_f_properties_l3210_321033

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 4 + 2 * Real.sin x * Real.cos x - Real.sin x ^ 4

theorem f_properties :
  (∀ x : ℝ, f (-x) ≠ f x ∧ f (-x) ≠ -f x) ∧
  (∀ ε > 0, ∃ x : ℝ, x > 0 ∧ x < π + ε ∧ f (x + π) = f x) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (- 3 * Real.pi / 8 + k * Real.pi) (k * Real.pi + Real.pi / 8))) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = Real.sqrt 2 ∧ ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f x) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -1 ∧ ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≥ f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3210_321033


namespace NUMINAMATH_CALUDE_cosine_sine_graph_shift_l3210_321073

theorem cosine_sine_graph_shift (x : ℝ) :
  3 * Real.cos (2 * x) = 3 * Real.sin (2 * (x + π / 6) + π / 6) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_graph_shift_l3210_321073


namespace NUMINAMATH_CALUDE_cab_driver_income_l3210_321070

theorem cab_driver_income (day2 day3 day4 day5 average : ℝ) 
  (h1 : day2 = 400)
  (h2 : day3 = 750)
  (h3 : day4 = 400)
  (h4 : day5 = 500)
  (h5 : average = 460)
  (h6 : average = (day1 + day2 + day3 + day4 + day5) / 5) :
  day1 = 250 := by
  sorry

#check cab_driver_income

end NUMINAMATH_CALUDE_cab_driver_income_l3210_321070


namespace NUMINAMATH_CALUDE_max_value_of_sqrt_sum_l3210_321086

theorem max_value_of_sqrt_sum (x : ℝ) (h : -9 ≤ x ∧ x ≤ 9) : 
  ∃ (max : ℝ), max = 6 ∧ 
  (∀ y : ℝ, -9 ≤ y ∧ y ≤ 9 → Real.sqrt (9 + y) + Real.sqrt (9 - y) ≤ max) ∧
  (∃ z : ℝ, -9 ≤ z ∧ z ≤ 9 ∧ Real.sqrt (9 + z) + Real.sqrt (9 - z) = max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sqrt_sum_l3210_321086


namespace NUMINAMATH_CALUDE_circle_reflection_and_translation_l3210_321083

def reflect_across_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

theorem circle_reflection_and_translation :
  let initial_center : ℝ × ℝ := (-3, -4)
  let reflected_center := reflect_across_x_axis initial_center
  let final_center := translate_up reflected_center 3
  final_center = (-3, 7) := by sorry

end NUMINAMATH_CALUDE_circle_reflection_and_translation_l3210_321083


namespace NUMINAMATH_CALUDE_function_equation_solution_l3210_321069

/-- Given functions f and g satisfying the condition for all x and y, 
    prove that f and g have the specified forms. -/
theorem function_equation_solution 
  (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f y + g x - g y = Real.sin x + Real.cos y) :
  (∃ c : ℝ, (∀ x : ℝ, f x = (Real.sin x + Real.cos x) / 2 ∧ 
                       g x = (Real.sin x - Real.cos x) / 2 + c)) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l3210_321069
