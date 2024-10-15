import Mathlib

namespace NUMINAMATH_CALUDE_largest_α_is_173_l1449_144903

/-- A triangle with angles satisfying specific conditions -/
structure SpecialTriangle where
  α : ℕ
  β : ℕ
  γ : ℕ
  angle_sum : α + β + γ = 180
  angle_order : α > β ∧ β > γ
  α_obtuse : α > 90
  α_prime : Nat.Prime α
  β_prime : Nat.Prime β

/-- The largest possible value of α in a SpecialTriangle is 173 -/
theorem largest_α_is_173 : ∀ t : SpecialTriangle, t.α ≤ 173 ∧ ∃ t' : SpecialTriangle, t'.α = 173 :=
  sorry

end NUMINAMATH_CALUDE_largest_α_is_173_l1449_144903


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1449_144929

theorem simplify_square_roots : Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^3) = 225 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1449_144929


namespace NUMINAMATH_CALUDE_equation_roots_l1449_144910

-- Define the equation
def equation (x : ℝ) : Prop :=
  (3*x^2 + 1)/(x-2) - (3*x+8)/4 + (5-9*x)/(x-2) + 2 = 0

-- Define the roots
def root1 : ℝ := 3.29
def root2 : ℝ := -0.40

-- Theorem statement
theorem equation_roots :
  ∃ (ε : ℝ), ε > 0 ∧ 
  (∀ (x : ℝ), equation x → (|x - root1| < ε ∨ |x - root2| < ε)) :=
sorry

end NUMINAMATH_CALUDE_equation_roots_l1449_144910


namespace NUMINAMATH_CALUDE_final_staff_count_l1449_144996

/- Define the initial number of staff in each category -/
def initial_doctors : ℕ := 11
def initial_nurses : ℕ := 18
def initial_assistants : ℕ := 9
def initial_interns : ℕ := 6

/- Define the number of staff who quit or are transferred -/
def doctors_quit : ℕ := 5
def nurses_quit : ℕ := 2
def assistants_quit : ℕ := 3
def nurses_transferred : ℕ := 2
def interns_transferred : ℕ := 4

/- Define the number of staff on leave -/
def doctors_on_leave : ℕ := 4
def nurses_on_leave : ℕ := 3

/- Define the number of new staff joining -/
def new_doctors : ℕ := 3
def new_nurses : ℕ := 5

/- Theorem to prove the final staff count -/
theorem final_staff_count :
  (initial_doctors - doctors_quit - doctors_on_leave + new_doctors) +
  (initial_nurses - nurses_quit - nurses_transferred - nurses_on_leave + new_nurses) +
  (initial_assistants - assistants_quit) +
  (initial_interns - interns_transferred) = 29 := by
  sorry

end NUMINAMATH_CALUDE_final_staff_count_l1449_144996


namespace NUMINAMATH_CALUDE_t_range_l1449_144963

/-- The function f(x) = |xe^x| -/
noncomputable def f (x : ℝ) : ℝ := |x * Real.exp x|

/-- The function g(x) = [f(x)]^2 - tf(x) -/
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := (f x)^2 - t * (f x)

/-- The theorem stating the range of t -/
theorem t_range (t : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    g t x₁ = -2 ∧ g t x₂ = -2 ∧ g t x₃ = -2 ∧ g t x₄ = -2) →
  t > Real.exp (-1) + 2 * Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_t_range_l1449_144963


namespace NUMINAMATH_CALUDE_multiplication_subtraction_difference_l1449_144937

theorem multiplication_subtraction_difference (x : ℝ) (h : x = 10) : 3 * x - (20 - x) = 20 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_difference_l1449_144937


namespace NUMINAMATH_CALUDE_series_sum_l1449_144934

/-- The general term of the series -/
def a (n : ℕ) : ℚ := (3 * n^2 + 2 * n + 1) / (n * (n + 1) * (n + 2) * (n + 3))

/-- The series sum -/
noncomputable def S : ℚ := ∑' n, a n

/-- Theorem: The sum of the series is 7/6 -/
theorem series_sum : S = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l1449_144934


namespace NUMINAMATH_CALUDE_total_salary_formula_l1449_144945

/-- Represents the total annual salary (in ten thousand yuan) paid by the enterprise in the nth year -/
def total_salary (n : ℕ) : ℝ :=
  (3 * n + 5) * (1.2 : ℝ)^n + 2.4

/-- The initial number of workers -/
def initial_workers : ℕ := 8

/-- The initial annual salary per worker (in yuan) -/
def initial_salary : ℝ := 10000

/-- The annual salary increase rate -/
def salary_increase_rate : ℝ := 0.2

/-- The number of new workers added each year -/
def new_workers_per_year : ℕ := 3

/-- The first-year salary of new workers (in yuan) -/
def new_worker_salary : ℝ := 8000

theorem total_salary_formula (n : ℕ) :
  total_salary n = (3 * n + initial_workers - 3) * (1 + salary_increase_rate)^n +
    (new_workers_per_year * new_worker_salary / 10000) := by
  sorry

end NUMINAMATH_CALUDE_total_salary_formula_l1449_144945


namespace NUMINAMATH_CALUDE_triangle_side_a_equals_one_l1449_144907

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (-Real.cos x, Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ :=
  let m_vec := m x
  let n_vec := n x
  (m_vec.1 * (0.5 * m_vec.1 - n_vec.1)) + (m_vec.2 * (0.5 * m_vec.2 - n_vec.2))

theorem triangle_side_a_equals_one (A B C : ℝ) (a b c : ℝ) :
  f (B / 2) = 1 → b = 1 → c = Real.sqrt 3 →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_a_equals_one_l1449_144907


namespace NUMINAMATH_CALUDE_modulus_of_z_l1449_144971

-- Define the complex number z
variable (z : ℂ)

-- Define the condition z(1-i) = 2i
def condition : Prop := z * (1 - Complex.I) = 2 * Complex.I

-- Theorem statement
theorem modulus_of_z (h : condition z) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1449_144971


namespace NUMINAMATH_CALUDE_percentage_of_360_l1449_144938

theorem percentage_of_360 : (33 + 1 / 3 : ℚ) / 100 * 360 = 120 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_l1449_144938


namespace NUMINAMATH_CALUDE_vector_equality_exists_l1449_144928

theorem vector_equality_exists (a b : ℝ × ℝ) :
  let a : ℝ × ℝ := (1, Real.sqrt 3)
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧
    let b : ℝ × ℝ := (Real.cos θ, Real.sin θ)
    ‖a + b‖ = ‖a - b‖ :=
by sorry

end NUMINAMATH_CALUDE_vector_equality_exists_l1449_144928


namespace NUMINAMATH_CALUDE_multiply_and_subtract_problem_solution_l1449_144984

theorem multiply_and_subtract (a b c : ℕ) : a * c - b * c = (a - b) * c := by sorry

theorem problem_solution : 65 * 1313 - 25 * 1313 = 52520 := by sorry

end NUMINAMATH_CALUDE_multiply_and_subtract_problem_solution_l1449_144984


namespace NUMINAMATH_CALUDE_empty_bucket_weight_l1449_144956

theorem empty_bucket_weight (full_weight : ℝ) (partial_weight : ℝ) : 
  full_weight = 3.4 →
  partial_weight = 2.98 →
  ∃ (empty_weight : ℝ),
    empty_weight = 1.3 ∧
    full_weight = empty_weight + (3.4 - empty_weight) ∧
    partial_weight = empty_weight + 4/5 * (3.4 - empty_weight) := by
  sorry

end NUMINAMATH_CALUDE_empty_bucket_weight_l1449_144956


namespace NUMINAMATH_CALUDE_f_five_l1449_144952

/-- A function satisfying f(xy) = 3xf(y) for all real x and y, with f(1) = 10 -/
def f : ℝ → ℝ :=
  sorry

/-- The functional equation for f -/
axiom f_eq (x y : ℝ) : f (x * y) = 3 * x * f y

/-- The value of f at 1 -/
axiom f_one : f 1 = 10

/-- The main theorem: f(5) = 150 -/
theorem f_five : f 5 = 150 := by
  sorry

end NUMINAMATH_CALUDE_f_five_l1449_144952


namespace NUMINAMATH_CALUDE_increasing_cubic_function_parameter_range_l1449_144924

theorem increasing_cubic_function_parameter_range 
  (f : ℝ → ℝ) (a : ℝ) (h : ∀ x ∈ Set.Ioo (-1) 1, StrictMono f) : a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_parameter_range_l1449_144924


namespace NUMINAMATH_CALUDE_unique_a_value_l1449_144985

/-- The base-72 number 235935623 -/
def base_72_num : ℕ := 235935623

/-- The proposition that the given base-72 number minus a is divisible by 9 -/
def is_divisible_by_nine (a : ℤ) : Prop :=
  (base_72_num : ℤ) - a ≡ 0 [ZMOD 9]

theorem unique_a_value :
  ∃! a : ℤ, 0 ≤ a ∧ a ≤ 18 ∧ is_divisible_by_nine a ∧ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1449_144985


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l1449_144947

theorem coefficient_x_cubed_in_binomial_expansion : 
  let n : ℕ := 5
  let a : ℝ := 1
  let b : ℝ := 2
  let r : ℕ := 3
  let coeff : ℝ := (n.choose r) * a^(n-r) * b^r
  coeff = 80 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l1449_144947


namespace NUMINAMATH_CALUDE_tank_capacity_l1449_144917

theorem tank_capacity (initial_fill : Rat) (added_amount : Rat) (final_fill : Rat) :
  initial_fill = 3 / 4 →
  added_amount = 8 →
  final_fill = 9 / 10 →
  ∃ (capacity : Rat), capacity = 160 / 3 ∧ 
    final_fill * capacity - initial_fill * capacity = added_amount :=
by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1449_144917


namespace NUMINAMATH_CALUDE_odds_against_C_l1449_144959

-- Define the type for horses
inductive Horse : Type
  | A
  | B
  | C

-- Define the race with no ties
def Race := Horse → ℕ

-- Define the odds against winning for each horse
def oddsAgainst (h : Horse) : ℚ :=
  match h with
  | Horse.A => 5/2
  | Horse.B => 3/1
  | Horse.C => 15/13  -- This is what we want to prove

-- Define the probability of winning for a horse given its odds against
def probWinning (odds : ℚ) : ℚ := 1 / (1 + odds)

-- State the theorem
theorem odds_against_C (race : Race) :
  (oddsAgainst Horse.A = 5/2) →
  (oddsAgainst Horse.B = 3/1) →
  (probWinning (oddsAgainst Horse.A) + probWinning (oddsAgainst Horse.B) + probWinning (oddsAgainst Horse.C) = 1) →
  oddsAgainst Horse.C = 15/13 := by
  sorry

end NUMINAMATH_CALUDE_odds_against_C_l1449_144959


namespace NUMINAMATH_CALUDE_subtracted_value_l1449_144977

theorem subtracted_value (N : ℝ) (x : ℝ) : 
  ((N - x) / 7 = 7) ∧ ((N - 2) / 13 = 4) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l1449_144977


namespace NUMINAMATH_CALUDE_pigeonhole_principle_balls_l1449_144940

theorem pigeonhole_principle_balls (red yellow blue : ℕ) :
  red > 0 ∧ yellow > 0 ∧ blue > 0 →
  ∃ n : ℕ, n = 4 ∧
    ∀ k : ℕ, k < n →
      ∃ f : Fin k → Fin 3,
        ∀ i j : Fin k, i ≠ j → f i = f j →
          ∃ m : ℕ, m ≥ n ∧
            ∀ g : Fin m → Fin 3,
              ∃ i j : Fin m, i ≠ j ∧ g i = g j :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_principle_balls_l1449_144940


namespace NUMINAMATH_CALUDE_winwin_processing_fee_l1449_144998

/-- Calculates the processing fee for a lottery win -/
def processing_fee (total_win : ℝ) (tax_rate : ℝ) (take_home : ℝ) : ℝ :=
  total_win * (1 - tax_rate) - take_home

/-- Theorem: The processing fee for Winwin's lottery win is $5 -/
theorem winwin_processing_fee :
  processing_fee 50 0.2 35 = 5 := by
  sorry

end NUMINAMATH_CALUDE_winwin_processing_fee_l1449_144998


namespace NUMINAMATH_CALUDE_inverse_of_ln_l1449_144923

theorem inverse_of_ln (x : ℝ) : 
  (fun y ↦ Real.exp y) ∘ (fun x ↦ Real.log x) = id ∧ 
  (fun x ↦ Real.log x) ∘ (fun y ↦ Real.exp y) = id :=
sorry

end NUMINAMATH_CALUDE_inverse_of_ln_l1449_144923


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l1449_144921

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l1449_144921


namespace NUMINAMATH_CALUDE_missing_number_proof_l1449_144978

def set1_sum (x y : ℝ) : ℝ := x + 50 + 78 + 104 + y
def set2_sum (x : ℝ) : ℝ := 48 + 62 + 98 + 124 + x

theorem missing_number_proof (x y : ℝ) :
  set1_sum x y / 5 = 62 ∧ set2_sum x / 5 = 76.4 → y = 28 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1449_144978


namespace NUMINAMATH_CALUDE_function_existence_condition_l1449_144989

theorem function_existence_condition (k a : ℕ) :
  (∃ f : ℕ → ℕ, ∀ n : ℕ, (f^[k] n) = n + a) ↔ (a = 0 ∨ a > 0) ∧ k ∣ a :=
by sorry

end NUMINAMATH_CALUDE_function_existence_condition_l1449_144989


namespace NUMINAMATH_CALUDE_half_jar_days_l1449_144954

/-- Represents the area of kombucha in the jar as a function of time -/
def kombucha_area (t : ℕ) : ℝ := 2^t

/-- The number of days it takes to fill the entire jar -/
def full_jar_days : ℕ := 17

theorem half_jar_days : 
  (kombucha_area full_jar_days = 2 * kombucha_area (full_jar_days - 1)) → 
  (kombucha_area (full_jar_days - 1) = (1/2) * kombucha_area full_jar_days) := by
  sorry

end NUMINAMATH_CALUDE_half_jar_days_l1449_144954


namespace NUMINAMATH_CALUDE_kangaroo_koala_ratio_l1449_144939

theorem kangaroo_koala_ratio :
  let total_animals : ℕ := 216
  let num_kangaroos : ℕ := 180
  let num_koalas : ℕ := total_animals - num_kangaroos
  num_kangaroos / num_koalas = 5 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_koala_ratio_l1449_144939


namespace NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l1449_144979

theorem richmond_tigers_ticket_sales (total_tickets second_half_tickets : ℕ) 
    (h1 : total_tickets = 9570)
    (h2 : second_half_tickets = 5703) :
  total_tickets - second_half_tickets = 3867 := by
  sorry

end NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l1449_144979


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l1449_144975

theorem quadratic_equation_root (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ x = 1) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l1449_144975


namespace NUMINAMATH_CALUDE_fraction_of_married_women_l1449_144901

/-- Given a company with employees, prove that 3/4 of women are married under specific conditions -/
theorem fraction_of_married_women (total : ℕ) (h_total_pos : total > 0) : 
  let women := (64 : ℚ) / 100 * total
  let married := (60 : ℚ) / 100 * total
  let men := total - women
  let single_men := (2 : ℚ) / 3 * men
  let married_men := men - single_men
  let married_women := married - married_men
  married_women / women = (3 : ℚ) / 4 := by
  sorry


end NUMINAMATH_CALUDE_fraction_of_married_women_l1449_144901


namespace NUMINAMATH_CALUDE_remainder_theorem_l1449_144908

-- Define the polynomial p(x)
def p (A B C : ℝ) (x : ℝ) : ℝ := A * x^5 + B * x^3 + C * x + 4

-- Theorem statement
theorem remainder_theorem (A B C : ℝ) :
  (p A B C 3 = 11) → (p A B C (-3) = -3) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1449_144908


namespace NUMINAMATH_CALUDE_fishing_competition_l1449_144987

/-- Fishing competition problem -/
theorem fishing_competition 
  (days : ℕ) 
  (jackson_per_day : ℕ) 
  (jonah_per_day : ℕ) 
  (total_catch : ℕ) :
  days = 5 →
  jackson_per_day = 6 →
  jonah_per_day = 4 →
  total_catch = 90 →
  ∃ (george_per_day : ℕ), 
    george_per_day = 8 ∧ 
    days * (jackson_per_day + jonah_per_day + george_per_day) = total_catch :=
by sorry

end NUMINAMATH_CALUDE_fishing_competition_l1449_144987


namespace NUMINAMATH_CALUDE_line_plane_relationship_l1449_144915

-- Define the types for lines and planes
variable (L P : Type*)

-- Define the perpendicular relationship between lines
variable (perp_line : L → L → Prop)

-- Define the perpendicular relationship between a line and a plane
variable (perp_line_plane : L → P → Prop)

-- Define the parallel relationship between a line and a plane
variable (parallel : L → P → Prop)

-- Define the subset relationship between a line and a plane
variable (subset : L → P → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : L) (α : P)
  (h1 : perp_line a b)
  (h2 : perp_line_plane b α) :
  subset a α ∨ parallel a α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l1449_144915


namespace NUMINAMATH_CALUDE_rectangle_area_l1449_144900

theorem rectangle_area (length width : ℚ) (h1 : length = 1/3) (h2 : width = 1/5) :
  length * width = 1/15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1449_144900


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l1449_144968

/-- 
Given a journey with total distance D and total time T, 
where a person travels 2/3 of D in 1/3 of T at 40 km/h,
prove that they must travel at 10 km/h for the remaining distance
to reach the destination on time.
-/
theorem journey_speed_calculation 
  (D T : ℝ) 
  (h1 : D > 0) 
  (h2 : T > 0) 
  (h3 : (2/3 * D) / (1/3 * T) = 40) : 
  (1/3 * D) / (2/3 * T) = 10 := by
sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l1449_144968


namespace NUMINAMATH_CALUDE_determinant_sin_matrix_l1449_144950

theorem determinant_sin_matrix (a b : Real) : 
  Matrix.det !![1, Real.sin (a - b), Real.sin a; 
                 Real.sin (a - b), 1, Real.sin b; 
                 Real.sin a, Real.sin b, 1] = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_sin_matrix_l1449_144950


namespace NUMINAMATH_CALUDE_linear_system_integer_solution_l1449_144992

theorem linear_system_integer_solution :
  ∃ (x y : ℤ), x + y = 5 ∧ 2 * x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_integer_solution_l1449_144992


namespace NUMINAMATH_CALUDE_unique_intersection_l1449_144943

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the property of having domain [-1, 5]
def HasDomain (f : RealFunction) : Prop :=
  ∀ x, x ∈ Set.Icc (-1) 5 → ∃ y, f x = y

-- Define the intersection of f with the line x=1
def Intersection (f : RealFunction) : Set ℝ :=
  {y : ℝ | f 1 = y}

-- Theorem statement
theorem unique_intersection
  (f : RealFunction) (h : HasDomain f) :
  ∃! y, y ∈ Intersection f :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l1449_144943


namespace NUMINAMATH_CALUDE_roberto_chicken_investment_break_even_l1449_144933

/-- Represents Roberto's chicken investment scenario -/
structure ChickenInvestment where
  num_chickens : ℕ
  cost_per_chicken : ℕ
  weekly_feed_cost : ℕ
  eggs_per_chicken_per_week : ℕ
  previous_dozen_cost : ℕ

/-- Calculates the break-even point in weeks for the chicken investment -/
def break_even_point (ci : ChickenInvestment) : ℕ :=
  let initial_cost := ci.num_chickens * ci.cost_per_chicken
  let weekly_egg_production := ci.num_chickens * ci.eggs_per_chicken_per_week
  let weekly_savings := ci.previous_dozen_cost - ci.weekly_feed_cost
  initial_cost / weekly_savings + 1

/-- Theorem stating that Roberto's chicken investment breaks even after 81 weeks -/
theorem roberto_chicken_investment_break_even :
  let ci : ChickenInvestment := {
    num_chickens := 4,
    cost_per_chicken := 20,
    weekly_feed_cost := 1,
    eggs_per_chicken_per_week := 3,
    previous_dozen_cost := 2
  }
  break_even_point ci = 81 := by sorry

end NUMINAMATH_CALUDE_roberto_chicken_investment_break_even_l1449_144933


namespace NUMINAMATH_CALUDE_boat_travel_time_l1449_144997

theorem boat_travel_time (boat_speed : ℝ) (distance : ℝ) (return_time : ℝ) :
  boat_speed = 15.6 →
  distance = 96 →
  return_time = 5 →
  ∃ (current_speed : ℝ),
    current_speed > 0 ∧
    current_speed < boat_speed ∧
    distance = (boat_speed + current_speed) * return_time ∧
    distance / (boat_speed - current_speed) = 8 :=
by sorry

end NUMINAMATH_CALUDE_boat_travel_time_l1449_144997


namespace NUMINAMATH_CALUDE_cab_delay_l1449_144995

theorem cab_delay (usual_time : ℝ) (speed_ratio : ℝ) (h1 : usual_time = 25) (h2 : speed_ratio = 5/6) :
  (usual_time / speed_ratio) - usual_time = 5 :=
by sorry

end NUMINAMATH_CALUDE_cab_delay_l1449_144995


namespace NUMINAMATH_CALUDE_paper_thickness_after_two_folds_l1449_144932

/-- The thickness of a paper after folding it in half a given number of times. -/
def thickness (initial : ℝ) (folds : ℕ) : ℝ :=
  initial * (2 ^ folds)

/-- Theorem: The thickness of a paper with initial thickness 0.1 mm after 2 folds is 0.4 mm. -/
theorem paper_thickness_after_two_folds :
  thickness 0.1 2 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_paper_thickness_after_two_folds_l1449_144932


namespace NUMINAMATH_CALUDE_cattle_breeder_milk_production_l1449_144902

/-- Calculates the weekly milk production for a given number of cows and daily milk production per cow. -/
def weekly_milk_production (num_cows : ℕ) (daily_production : ℕ) : ℕ :=
  num_cows * daily_production * 7

/-- Proves that the weekly milk production of 52 cows, each producing 1000 oz of milk per day, is 364,000 oz. -/
theorem cattle_breeder_milk_production :
  weekly_milk_production 52 1000 = 364000 := by
  sorry


end NUMINAMATH_CALUDE_cattle_breeder_milk_production_l1449_144902


namespace NUMINAMATH_CALUDE_steven_extra_seeds_l1449_144944

/-- Represents the number of seeds in different fruits -/
structure FruitSeeds where
  apple : Nat
  pear : Nat
  grape : Nat
  orange : Nat
  watermelon : Nat

/-- Represents the number of each fruit Steven has -/
structure StevenFruits where
  apples : Nat
  pears : Nat
  grapes : Nat
  oranges : Nat
  watermelons : Nat

def required_seeds : Nat := 420

def average_seeds : FruitSeeds := {
  apple := 6,
  pear := 2,
  grape := 3,
  orange := 10,
  watermelon := 300
}

def steven_fruits : StevenFruits := {
  apples := 2,
  pears := 3,
  grapes := 5,
  oranges := 1,
  watermelons := 2
}

/-- Calculates the total number of seeds Steven has -/
def total_seeds (avg : FruitSeeds) (fruits : StevenFruits) : Nat :=
  avg.apple * fruits.apples +
  avg.pear * fruits.pears +
  avg.grape * fruits.grapes +
  avg.orange * fruits.oranges +
  avg.watermelon * fruits.watermelons

/-- Theorem stating that Steven has 223 more seeds than required -/
theorem steven_extra_seeds :
  total_seeds average_seeds steven_fruits - required_seeds = 223 := by
  sorry

end NUMINAMATH_CALUDE_steven_extra_seeds_l1449_144944


namespace NUMINAMATH_CALUDE_probability_intersecting_diagonals_l1449_144988

/-- A regular decagon -/
structure RegularDecagon where
  -- Add any necessary properties

/-- Represents a diagonal in a regular decagon -/
structure Diagonal where
  -- Add any necessary properties

/-- The set of all diagonals in a regular decagon -/
def allDiagonals (d : RegularDecagon) : Set Diagonal :=
  sorry

/-- Predicate to check if two diagonals intersect inside the decagon -/
def intersectInside (d : RegularDecagon) (d1 d2 : Diagonal) : Prop :=
  sorry

/-- The number of ways to choose 3 diagonals from all diagonals -/
def numWaysChoose3Diagonals (d : RegularDecagon) : ℕ :=
  sorry

/-- The number of ways to choose 3 diagonals where at least two intersect -/
def numWaysChoose3IntersectingDiagonals (d : RegularDecagon) : ℕ :=
  sorry

theorem probability_intersecting_diagonals (d : RegularDecagon) :
    (numWaysChoose3IntersectingDiagonals d : ℚ) / (numWaysChoose3Diagonals d : ℚ) = 252 / 1309 := by
  sorry

end NUMINAMATH_CALUDE_probability_intersecting_diagonals_l1449_144988


namespace NUMINAMATH_CALUDE_solution_set_and_range_l1449_144949

def f (x : ℝ) : ℝ := |x + 1|

theorem solution_set_and_range :
  (∀ x : ℝ, f x ≤ 5 - f (x - 3) ↔ -2 ≤ x ∧ x ≤ 3) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * f x + |x + a| ≤ x + 4 → -2 ≤ a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l1449_144949


namespace NUMINAMATH_CALUDE_hour_division_theorem_l1449_144925

/-- The number of seconds in an hour -/
def seconds_in_hour : ℕ := 3600

/-- The number of ways to divide an hour into periods -/
def num_divisions : ℕ := 44

/-- Theorem: The number of ordered pairs of positive integers (n, m) 
    such that n * m = 3600 is equal to 44 -/
theorem hour_division_theorem : 
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = seconds_in_hour) 
    (Finset.product (Finset.range (seconds_in_hour + 1)) (Finset.range (seconds_in_hour + 1)))).card 
  = num_divisions := by
  sorry


end NUMINAMATH_CALUDE_hour_division_theorem_l1449_144925


namespace NUMINAMATH_CALUDE_profit_calculation_l1449_144914

/-- The number of pens John buys for $8 -/
def pens_bought : ℕ := 5

/-- The price John pays for pens_bought pens -/
def buy_price : ℚ := 8

/-- The number of pens John sells for $10 -/
def pens_sold : ℕ := 4

/-- The price John receives for pens_sold pens -/
def sell_price : ℚ := 10

/-- The desired profit -/
def target_profit : ℚ := 120

/-- The minimum number of pens John needs to sell to make the target profit -/
def min_pens_to_sell : ℕ := 134

theorem profit_calculation :
  ↑min_pens_to_sell * (sell_price / pens_sold - buy_price / pens_bought) ≥ target_profit ∧
  ∀ n : ℕ, n < min_pens_to_sell → ↑n * (sell_price / pens_sold - buy_price / pens_bought) < target_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_calculation_l1449_144914


namespace NUMINAMATH_CALUDE_dust_storm_untouched_acres_l1449_144981

/-- The number of acres left untouched by a dust storm -/
def acres_untouched (total_acres dust_covered_acres : ℕ) : ℕ :=
  total_acres - dust_covered_acres

/-- Theorem stating that given a prairie of 65,057 acres and a dust storm covering 64,535 acres, 
    the number of acres left untouched is 522 -/
theorem dust_storm_untouched_acres : 
  acres_untouched 65057 64535 = 522 := by
  sorry

end NUMINAMATH_CALUDE_dust_storm_untouched_acres_l1449_144981


namespace NUMINAMATH_CALUDE_todds_contribution_ratio_l1449_144953

theorem todds_contribution_ratio (total_cost : ℕ) (boss_contribution : ℕ) 
  (num_employees : ℕ) (employee_contribution : ℕ) : 
  total_cost = 100 →
  boss_contribution = 15 →
  num_employees = 5 →
  employee_contribution = 11 →
  (total_cost - (boss_contribution + num_employees * employee_contribution)) / boss_contribution = 2 := by
  sorry

end NUMINAMATH_CALUDE_todds_contribution_ratio_l1449_144953


namespace NUMINAMATH_CALUDE_point_m_property_l1449_144926

theorem point_m_property (a : ℝ) : 
  let m : ℝ × ℝ := (3*a - 9, 10 - 2*a)
  (m.1 < 0 ∧ m.2 > 0) →  -- M is in the second quadrant
  (|m.1| = |m.2|) →      -- Distance to y-axis equals distance to x-axis
  (a + 2)^2023 - 1 = 0   -- The expression equals 0
  := by sorry

end NUMINAMATH_CALUDE_point_m_property_l1449_144926


namespace NUMINAMATH_CALUDE_solve_for_y_l1449_144965

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 4*x = y + 5) (h2 : x = 7) : y = 16 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1449_144965


namespace NUMINAMATH_CALUDE_attic_junk_items_l1449_144990

theorem attic_junk_items (total : ℕ) (useful : ℕ) (junk_percent : ℚ) :
  useful = (20 : ℚ) / 100 * total →
  junk_percent = 70 / 100 →
  useful = 8 →
  ⌊junk_percent * total⌋ = 28 := by
sorry

end NUMINAMATH_CALUDE_attic_junk_items_l1449_144990


namespace NUMINAMATH_CALUDE_juniper_whiskers_l1449_144912

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ
  juniper : ℕ

/-- Defines the conditions for the cat whisker problem -/
def whisker_conditions (c : CatWhiskers) : Prop :=
  c.buffy = 40 ∧
  c.puffy = 3 * c.juniper ∧
  c.puffy = c.scruffy / 2 ∧
  c.buffy = (c.puffy + c.scruffy + c.juniper) / 3

/-- Theorem stating that under the given conditions, Juniper has 12 whiskers -/
theorem juniper_whiskers (c : CatWhiskers) : 
  whisker_conditions c → c.juniper = 12 := by
  sorry

#check juniper_whiskers

end NUMINAMATH_CALUDE_juniper_whiskers_l1449_144912


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l1449_144983

/-- Checks if a number is a palindrome in the given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∀ k : ℕ,
  k > 10 →
  (isPalindrome k 2 ∧ isPalindrome k 4) →
  k ≥ 17 ∧
  isPalindrome 17 2 ∧
  isPalindrome 17 4 := by
    sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l1449_144983


namespace NUMINAMATH_CALUDE_n_div_30_n_squared_cube_n_cubed_square_n_smallest_n_has_three_digits_l1449_144994

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
theorem n_div_30 : 30 ∣ n := sorry

/-- n^2 is a perfect cube -/
theorem n_squared_cube : ∃ k : ℕ, n^2 = k^3 := sorry

/-- n^3 is a perfect square -/
theorem n_cubed_square : ∃ k : ℕ, n^3 = k^2 := sorry

/-- n is the smallest positive integer satisfying the conditions -/
theorem n_smallest : ∀ m : ℕ, m < n → ¬(30 ∣ m ∧ (∃ k : ℕ, m^2 = k^3) ∧ (∃ k : ℕ, m^3 = k^2)) := sorry

/-- The number of digits in n -/
def digits_of_n : ℕ := sorry

/-- Theorem stating that n has 3 digits -/
theorem n_has_three_digits : digits_of_n = 3 := sorry

end NUMINAMATH_CALUDE_n_div_30_n_squared_cube_n_cubed_square_n_smallest_n_has_three_digits_l1449_144994


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l1449_144951

theorem min_x_prime_factorization (x y : ℕ+) (h : 5 * x^7 = 13 * y^17) :
  ∃ (a b c d : ℕ),
    x.val = a^c * b^d ∧
    a.Prime ∧ b.Prime ∧
    x.val ≥ 13^5 * 5^10 ∧
    (x.val = 13^5 * 5^10 → a + b + c + d = 33) :=
sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l1449_144951


namespace NUMINAMATH_CALUDE_car_speed_equation_l1449_144970

/-- Given a car traveling at speed v km/h, prove that v satisfies the equation
    v = 3600 / 49, if it takes 4 seconds longer to travel 1 km at speed v
    than at 80 km/h. -/
theorem car_speed_equation (v : ℝ) : v > 0 →
  (3600 / v = 3600 / 80 + 4) → v = 3600 / 49 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_equation_l1449_144970


namespace NUMINAMATH_CALUDE_younger_major_A_probability_l1449_144905

structure GraduatingClass where
  maleProportion : Real
  majorAProb : Real
  majorBProb : Real
  majorCProb : Real
  maleOlderProb : Real
  femaleOlderProb : Real
  majorAOlderProb : Real
  majorBOlderProb : Real
  majorCOlderProb : Real

def probabilityYoungerMajorA (gc : GraduatingClass) : Real :=
  gc.majorAProb * (1 - gc.majorAOlderProb)

theorem younger_major_A_probability (gc : GraduatingClass) 
  (h1 : gc.maleProportion = 0.4)
  (h2 : gc.majorAProb = 0.5)
  (h3 : gc.majorBProb = 0.3)
  (h4 : gc.majorCProb = 0.2)
  (h5 : gc.maleOlderProb = 0.5)
  (h6 : gc.femaleOlderProb = 0.3)
  (h7 : gc.majorAOlderProb = 0.6)
  (h8 : gc.majorBOlderProb = 0.4)
  (h9 : gc.majorCOlderProb = 0.2) :
  probabilityYoungerMajorA gc = 0.2 := by
  sorry

#check younger_major_A_probability

end NUMINAMATH_CALUDE_younger_major_A_probability_l1449_144905


namespace NUMINAMATH_CALUDE_triangle_area_prove_triangle_area_l1449_144918

/-- Parabola equation: x^2 = 16y -/
def parabola (x y : ℝ) : Prop := x^2 = 16 * y

/-- Hyperbola equation: x^2 - y^2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- Directrix of the parabola -/
def directrix : ℝ := -4

/-- Asymptotes of the hyperbola -/
def asymptote₁ (x : ℝ) : ℝ := x
def asymptote₂ (x : ℝ) : ℝ := -x

/-- Points where asymptotes intersect the directrix -/
def point₁ : ℝ × ℝ := (4, -4)
def point₂ : ℝ × ℝ := (-4, -4)

/-- The area of the triangle formed by the directrix and asymptotes -/
theorem triangle_area : ℝ := 16

/-- Proof that the area of the triangle is 16 -/
theorem prove_triangle_area : triangle_area = 16 := by sorry

end NUMINAMATH_CALUDE_triangle_area_prove_triangle_area_l1449_144918


namespace NUMINAMATH_CALUDE_xy_values_l1449_144957

theorem xy_values (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 62) :
  x * y = -126/25 ∨ x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_values_l1449_144957


namespace NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l1449_144942

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 68 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_5_balls_4_boxes : distribute_balls 5 4 = 68 := by sorry

end NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l1449_144942


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1449_144946

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16) →
  a 3 + a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1449_144946


namespace NUMINAMATH_CALUDE_group_abelian_l1449_144973

variable {G : Type*} [Group G]

theorem group_abelian (h : ∀ x : G, x * x = 1) : ∀ a b : G, a * b = b * a := by
  sorry

end NUMINAMATH_CALUDE_group_abelian_l1449_144973


namespace NUMINAMATH_CALUDE_set_relationship_l1449_144986

def M : Set ℤ := {x | ∃ k : ℤ, x = 5 * k - 2}
def P : Set ℤ := {x | ∃ n : ℤ, x = 5 * n + 3}
def S : Set ℤ := {x | ∃ m : ℤ, x = 10 * m + 3}

theorem set_relationship : S ⊆ P ∧ P = M := by sorry

end NUMINAMATH_CALUDE_set_relationship_l1449_144986


namespace NUMINAMATH_CALUDE_constant_fraction_iff_proportional_coefficients_l1449_144960

/-- A fraction of quadratic polynomials is constant if and only if the coefficients are proportional -/
theorem constant_fraction_iff_proportional_coefficients 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) (h : a₂ ≠ 0) :
  (∃ k : ℝ, ∀ x : ℝ, (a₁ * x^2 + b₁ * x + c₁) / (a₂ * x^2 + b₂ * x + c₂) = k) ↔ 
  (∃ k : ℝ, a₁ = k * a₂ ∧ b₁ = k * b₂ ∧ c₁ = k * c₂) :=
sorry

end NUMINAMATH_CALUDE_constant_fraction_iff_proportional_coefficients_l1449_144960


namespace NUMINAMATH_CALUDE_jensen_family_trip_l1449_144980

/-- Calculates the miles driven on city streets given the total distance on highways,
    car efficiency on highways and city streets, and total gas used. -/
theorem jensen_family_trip (highway_miles : ℝ) (highway_efficiency : ℝ) 
  (city_efficiency : ℝ) (total_gas : ℝ) (city_miles : ℝ) : 
  highway_miles = 210 →
  highway_efficiency = 35 →
  city_efficiency = 18 →
  total_gas = 9 →
  city_miles = (total_gas - highway_miles / highway_efficiency) * city_efficiency →
  city_miles = 54 := by sorry

end NUMINAMATH_CALUDE_jensen_family_trip_l1449_144980


namespace NUMINAMATH_CALUDE_frankie_candy_count_l1449_144966

theorem frankie_candy_count (max_candy : ℕ) (extra_candy : ℕ) (frankie_candy : ℕ) : 
  max_candy = 92 → 
  extra_candy = 18 → 
  max_candy = frankie_candy + extra_candy → 
  frankie_candy = 74 := by
sorry

end NUMINAMATH_CALUDE_frankie_candy_count_l1449_144966


namespace NUMINAMATH_CALUDE_pascal_triangle_elements_l1449_144920

/-- The number of elements in the nth row of Pascal's Triangle -/
def elementsInRow (n : ℕ) : ℕ := n + 1

/-- The sum of elements from row 0 to row n of Pascal's Triangle -/
def sumElements (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

theorem pascal_triangle_elements : sumElements 29 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_elements_l1449_144920


namespace NUMINAMATH_CALUDE_p_satisfies_conditions_l1449_144961

/-- A quadratic polynomial that satisfies specific conditions -/
def p (x : ℝ) : ℝ := -3 * x^2 - 9 * x + 84

/-- Theorem stating that p satisfies the required conditions -/
theorem p_satisfies_conditions :
  p (-7) = 0 ∧ p 4 = 0 ∧ p 5 = -36 := by
  sorry

end NUMINAMATH_CALUDE_p_satisfies_conditions_l1449_144961


namespace NUMINAMATH_CALUDE_special_sequence_has_large_number_l1449_144972

/-- A sequence of natural numbers with the given properties -/
def SpecialSequence (seq : Fin 20 → ℕ) : Prop :=
  (∀ i, seq i ≠ seq (i + 1)) ∧  -- distinct numbers
  (∀ i < 19, ∃ k : ℕ, seq i * seq (i + 1) = k * k) ∧  -- product is perfect square
  seq 0 = 42  -- first number is 42

theorem special_sequence_has_large_number (seq : Fin 20 → ℕ) 
  (h : SpecialSequence seq) : 
  ∃ i, seq i > 16000 := by
sorry

end NUMINAMATH_CALUDE_special_sequence_has_large_number_l1449_144972


namespace NUMINAMATH_CALUDE_power_sum_theorem_l1449_144967

theorem power_sum_theorem (a b x y : ℝ) 
  (h1 : a*x + b*y = 5)
  (h2 : a*x^2 + b*y^2 = 11)
  (h3 : a*x^3 + b*y^3 = 25)
  (h4 : a*x^4 + b*y^4 = 58) :
  a*x^5 + b*y^5 = 136.25 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_theorem_l1449_144967


namespace NUMINAMATH_CALUDE_percentage_subtracted_from_b_l1449_144969

theorem percentage_subtracted_from_b (a b x m : ℝ) (p : ℝ) : 
  a > 0 ∧ b > 0 ∧
  a / b = 4 / 5 ∧
  x = a + 0.75 * a ∧
  m = b - (p / 100) * b ∧
  m / x = 0.14285714285714285 →
  p = 80 := by
sorry

end NUMINAMATH_CALUDE_percentage_subtracted_from_b_l1449_144969


namespace NUMINAMATH_CALUDE_magazine_subscription_pigeonhole_l1449_144999

theorem magazine_subscription_pigeonhole 
  (total_students : Nat) 
  (subscription_combinations : Nat) 
  (h1 : total_students = 39) 
  (h2 : subscription_combinations = 7) :
  ∃ (combination : Nat), combination ≤ subscription_combinations ∧ 
    (total_students / subscription_combinations + 1 : Nat) ≤ 
      (λ i => (total_students / subscription_combinations : Nat) + 
        if i ≤ (total_students % subscription_combinations) then 1 else 0) combination :=
by
  sorry

end NUMINAMATH_CALUDE_magazine_subscription_pigeonhole_l1449_144999


namespace NUMINAMATH_CALUDE_park_animals_ratio_l1449_144935

theorem park_animals_ratio (lions leopards elephants : ℕ) : 
  lions = 200 →
  elephants = (lions + leopards) / 2 →
  lions + leopards + elephants = 450 →
  lions = 2 * leopards :=
by
  sorry

end NUMINAMATH_CALUDE_park_animals_ratio_l1449_144935


namespace NUMINAMATH_CALUDE_rhombus_diagonals_not_necessarily_equal_l1449_144916

/-- Definition of a rhombus -/
structure Rhombus :=
  (sides : Fin 4 → ℝ)
  (equal_sides : ∀ i j : Fin 4, sides i = sides j)
  (perpendicular_diagonals : True)  -- We simplify this condition for the purpose of this problem

/-- Definition of diagonals of a rhombus -/
def diagonals (r : Rhombus) : Fin 2 → ℝ := sorry

/-- Theorem stating that the diagonals of a rhombus are not necessarily equal -/
theorem rhombus_diagonals_not_necessarily_equal :
  ¬ (∀ r : Rhombus, diagonals r 0 = diagonals r 1) :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_not_necessarily_equal_l1449_144916


namespace NUMINAMATH_CALUDE_reflection_segment_length_d_to_d_prime_length_l1449_144976

/-- The length of the segment from a point to its reflection over the x-axis --/
theorem reflection_segment_length (x y : ℝ) : 
  let d : ℝ × ℝ := (x, y)
  let d_reflected : ℝ × ℝ := (x, -y)
  Real.sqrt ((d.1 - d_reflected.1)^2 + (d.2 - d_reflected.2)^2) = 2 * |y| :=
by sorry

/-- The length of the segment from D(-5, 3) to its reflection D' over the x-axis is 6 --/
theorem d_to_d_prime_length : 
  let d : ℝ × ℝ := (-5, 3)
  let d_reflected : ℝ × ℝ := (-5, -3)
  Real.sqrt ((d.1 - d_reflected.1)^2 + (d.2 - d_reflected.2)^2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_reflection_segment_length_d_to_d_prime_length_l1449_144976


namespace NUMINAMATH_CALUDE_range_of_x_l1449_144958

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being even
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of f being decreasing on [0,+∞)
def IsDecreasingOnNonnegativeReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

-- State the theorem
theorem range_of_x (h1 : IsEven f) (h2 : IsDecreasingOnNonnegativeReals f) 
  (h3 : ∀ x > 0, f (Real.log x / Real.log 10) > f 1) :
  ∀ x > 0, f (Real.log x / Real.log 10) > f 1 → 1/10 < x ∧ x < 10 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1449_144958


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l1449_144911

theorem smallest_solution_floor_equation :
  (∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 12) ∧
  (∀ (y : ℝ), y > 0 ∧ ⌊y^2⌋ - y * ⌊y⌋ = 12 → y ≥ 169/13) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l1449_144911


namespace NUMINAMATH_CALUDE_ice_cream_cost_l1449_144991

theorem ice_cream_cost (price : ℚ) (discount : ℚ) : 
  price = 99/100 ∧ discount = 1/10 → 
  price + price * (1 - discount) = 1881/1000 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l1449_144991


namespace NUMINAMATH_CALUDE_exists_number_with_specific_digit_sum_l1449_144931

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with specific digit sum properties -/
theorem exists_number_with_specific_digit_sum : 
  ∃ m : ℕ, digit_sum m = 1990 ∧ digit_sum (m^2) = 1990^2 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_specific_digit_sum_l1449_144931


namespace NUMINAMATH_CALUDE_multinomial_expansion_terms_l1449_144904

/-- The number of terms in the simplified multinomial expansion of (x+y+z)^10 -/
def multinomial_terms : ℕ := 66

/-- Theorem stating that the number of terms in the simplified multinomial expansion of (x+y+z)^10 is 66 -/
theorem multinomial_expansion_terms :
  multinomial_terms = 66 := by sorry

end NUMINAMATH_CALUDE_multinomial_expansion_terms_l1449_144904


namespace NUMINAMATH_CALUDE_negation_of_square_positive_l1449_144922

theorem negation_of_square_positive :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, ¬(x^2 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_square_positive_l1449_144922


namespace NUMINAMATH_CALUDE_tom_hockey_games_l1449_144919

/-- The number of hockey games Tom attended this year -/
def games_this_year : ℕ := 4

/-- The number of hockey games Tom attended last year -/
def games_last_year : ℕ := 9

/-- The total number of hockey games Tom attended -/
def total_games : ℕ := games_this_year + games_last_year

theorem tom_hockey_games :
  total_games = 13 := by sorry

end NUMINAMATH_CALUDE_tom_hockey_games_l1449_144919


namespace NUMINAMATH_CALUDE_equation_always_has_real_root_l1449_144993

theorem equation_always_has_real_root :
  ∀ (q : ℝ), ∃ (x : ℝ), x^6 + q*x^4 + q^2*x^2 + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_always_has_real_root_l1449_144993


namespace NUMINAMATH_CALUDE_disjunction_true_l1449_144927

def p : Prop := ∃ k : ℕ, 2 = 2 * k

def q : Prop := ∃ k : ℕ, 3 = 2 * k

theorem disjunction_true : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_true_l1449_144927


namespace NUMINAMATH_CALUDE_triangle_inequality_ortho_segments_inequality_not_always_true_l1449_144962

/-- A triangle with sides a ≥ b ≥ c and corresponding altitudes m_a ≤ m_b ≤ m_c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  h_sides : a ≥ b ∧ b ≥ c
  h_altitudes : m_a ≤ m_b ∧ m_b ≤ m_c

/-- Lengths of segments from vertex to orthocenter along corresponding altitudes -/
structure OrthoSegments where
  m_a_star : ℝ
  m_b_star : ℝ
  m_c_star : ℝ

/-- Theorem stating the inequality for sides and altitudes -/
theorem triangle_inequality (t : Triangle) : t.a + t.m_a ≥ t.b + t.m_b ∧ t.b + t.m_b ≥ t.c + t.m_c :=
  sorry

/-- Statement that the inequality for orthocenter segments is not always true -/
theorem ortho_segments_inequality_not_always_true : 
  ¬ ∀ (t : Triangle) (o : OrthoSegments), t.a + o.m_a_star ≥ t.b + o.m_b_star ∧ t.b + o.m_b_star ≥ t.c + o.m_c_star :=
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_ortho_segments_inequality_not_always_true_l1449_144962


namespace NUMINAMATH_CALUDE_absolute_value_probability_l1449_144964

theorem absolute_value_probability (x : ℝ) : ℝ := by
  have h : ∀ x : ℝ, |x| ≥ 0 := by sorry
  have event : Set ℝ := {x | |x| < 0}
  have prob_event : ℝ := 0
  sorry

end NUMINAMATH_CALUDE_absolute_value_probability_l1449_144964


namespace NUMINAMATH_CALUDE_tom_uncommon_cards_l1449_144974

/-- Represents the deck composition and cost in Tom's trading card game. -/
structure DeckInfo where
  rare_count : ℕ
  common_count : ℕ
  rare_cost : ℚ
  uncommon_cost : ℚ
  common_cost : ℚ
  total_cost : ℚ

/-- Calculates the number of uncommon cards in the deck. -/
def uncommon_count (deck : DeckInfo) : ℕ :=
  let rare_total := deck.rare_count * deck.rare_cost
  let common_total := deck.common_count * deck.common_cost
  let uncommon_total := deck.total_cost - rare_total - common_total
  (uncommon_total / deck.uncommon_cost).num.toNat

/-- Theorem stating that Tom's deck contains 11 uncommon cards. -/
theorem tom_uncommon_cards : 
  let deck : DeckInfo := {
    rare_count := 19,
    common_count := 30,
    rare_cost := 1,
    uncommon_cost := 1/2,
    common_cost := 1/4,
    total_cost := 32
  }
  uncommon_count deck = 11 := by sorry

end NUMINAMATH_CALUDE_tom_uncommon_cards_l1449_144974


namespace NUMINAMATH_CALUDE_parabola_line_tangency_l1449_144982

/-- 
Given a parabola y = ax^2 + 6 and a line y = 2x + k, where k is a constant,
this theorem states the condition for tangency between the parabola and the line.
-/
theorem parabola_line_tangency (a k : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + 6 ∧ y = 2 * x + k) →
  (k ≠ 6) →
  (a = -1 / (k - 6)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_tangency_l1449_144982


namespace NUMINAMATH_CALUDE_inequality_proof_l1449_144913

theorem inequality_proof (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1449_144913


namespace NUMINAMATH_CALUDE_cube_root_of_a_plus_b_plus_one_l1449_144948

theorem cube_root_of_a_plus_b_plus_one (a b : ℝ) 
  (h1 : (2 * a - 1) = 9)
  (h2 : (3 * a + b - 1) = 16) : 
  (a + b + 1)^(1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_a_plus_b_plus_one_l1449_144948


namespace NUMINAMATH_CALUDE_cube_root_two_identity_l1449_144930

theorem cube_root_two_identity (x : ℝ) (h : 32 = x^6 + 1/x^6) :
  x^2 + 1/x^2 = 2 * Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_two_identity_l1449_144930


namespace NUMINAMATH_CALUDE_zero_multiple_of_all_integers_l1449_144941

theorem zero_multiple_of_all_integers : ∀ (n : ℤ), ∃ (k : ℤ), 0 = k * n := by
  sorry

end NUMINAMATH_CALUDE_zero_multiple_of_all_integers_l1449_144941


namespace NUMINAMATH_CALUDE_max_value_product_ratios_l1449_144909

/-- Line l in Cartesian coordinates -/
def line_l (y : ℝ) : Prop := y = 8

/-- Circle C in parametric form -/
def circle_C (x y φ : ℝ) : Prop := x = 2 + 2 * Real.cos φ ∧ y = 2 * Real.sin φ

/-- Ray OM in polar coordinates -/
def ray_OM (θ α : ℝ) : Prop := θ = α ∧ 0 < α ∧ α < Real.pi / 2

/-- Ray ON in polar coordinates -/
def ray_ON (θ α : ℝ) : Prop := θ = α - Real.pi / 2

/-- Theorem stating the maximum value of the product of ratios -/
theorem max_value_product_ratios (α : ℝ) 
  (h_ray_OM : ray_OM α α) 
  (h_ray_ON : ray_ON (α - Real.pi / 2) α) : 
  ∃ (OP OM OQ ON : ℝ), 
    (OP / OM) * (OQ / ON) ≤ 1 / 16 ∧ 
    ∃ (α_max : ℝ), (OP / OM) * (OQ / ON) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_product_ratios_l1449_144909


namespace NUMINAMATH_CALUDE_regular_polygon_30_degree_central_angle_has_12_sides_l1449_144906

/-- Theorem: A regular polygon with a central angle of 30° has 12 sides. -/
theorem regular_polygon_30_degree_central_angle_has_12_sides :
  ∀ (n : ℕ) (central_angle : ℝ),
    central_angle = 30 →
    (360 : ℝ) / central_angle = n →
    n = 12 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_30_degree_central_angle_has_12_sides_l1449_144906


namespace NUMINAMATH_CALUDE_original_number_l1449_144955

theorem original_number (t : ℝ) : 
  t * (1 + 0.125) - t * (1 - 0.25) = 30 → t = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1449_144955


namespace NUMINAMATH_CALUDE_f_monotonically_decreasing_l1449_144936

-- Define the function f(x) = e^x / x
noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

-- Define the property of being monotonically decreasing on an interval
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f y < f x

-- State the theorem
theorem f_monotonically_decreasing :
  (∀ x y, x < y → y < 0 → f y < f x) ∧
  (∀ x y, 0 < x → x < y → y ≤ 1 → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_f_monotonically_decreasing_l1449_144936
