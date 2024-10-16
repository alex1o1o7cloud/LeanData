import Mathlib

namespace NUMINAMATH_CALUDE_ant_problem_l187_18701

/-- Represents the position of an ant on a square path -/
structure AntPosition where
  side : ℕ  -- 0: bottom, 1: right, 2: top, 3: left
  distance : ℝ  -- distance from the start of the side

/-- Represents a square path -/
structure SquarePath where
  sideLength : ℝ

/-- Represents the state of the three ants -/
structure AntState where
  mu : AntPosition
  ra : AntPosition
  vey : AntPosition

/-- Checks if the ants are aligned on a straight line -/
def areAntsAligned (state : AntState) (paths : List SquarePath) : Prop :=
  sorry

/-- Updates the positions of the ants based on the distance they've traveled -/
def updateAntPositions (initialState : AntState) (paths : List SquarePath) (distance : ℝ) : AntState :=
  sorry

theorem ant_problem (a : ℝ) :
  let paths := [⟨a⟩, ⟨a + 2⟩, ⟨a + 4⟩]
  let initialState : AntState := {
    mu := { side := 0, distance := 0 },
    ra := { side := 0, distance := 0 },
    vey := { side := 0, distance := 0 }
  }
  let finalState := updateAntPositions initialState paths ((a + 4) / 2)
  finalState.mu.side = 1 ∧
  finalState.mu.distance = 0 ∧
  finalState.ra.side = 1 ∧
  finalState.vey.side = 1 ∧
  areAntsAligned finalState paths →
  a = 4 :=
sorry

end NUMINAMATH_CALUDE_ant_problem_l187_18701


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l187_18717

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  is_geometric a →
  (a 3)^2 + 7*(a 3) + 9 = 0 →
  (a 7)^2 + 7*(a 7) + 9 = 0 →
  (a 5 = 3 ∨ a 5 = -3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l187_18717


namespace NUMINAMATH_CALUDE_number_reduced_then_increased_l187_18718

theorem number_reduced_then_increased : ∃ x : ℝ, (20 * (x / 5) = 40) ∧ (x = 10) := by
  sorry

end NUMINAMATH_CALUDE_number_reduced_then_increased_l187_18718


namespace NUMINAMATH_CALUDE_sine_angle_equality_l187_18783

theorem sine_angle_equality (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * π / 180) = Real.sin (604 * π / 180) → n = -64 := by
  sorry

end NUMINAMATH_CALUDE_sine_angle_equality_l187_18783


namespace NUMINAMATH_CALUDE_negation_of_existence_l187_18713

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x, x^2 - x + 1 = 0) ↔ (∀ x, x^2 - x + 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l187_18713


namespace NUMINAMATH_CALUDE_gcd_1343_816_l187_18742

theorem gcd_1343_816 : Nat.gcd 1343 816 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1343_816_l187_18742


namespace NUMINAMATH_CALUDE_int_tan_triangle_unique_l187_18791

/-- A triangle with integer tangents for all angles -/
structure IntTanTriangle where
  α : Real
  β : Real
  γ : Real
  sum_180 : α + β + γ = Real.pi
  tan_int_α : ∃ m : Int, Real.tan α = m
  tan_int_β : ∃ n : Int, Real.tan β = n
  tan_int_γ : ∃ k : Int, Real.tan γ = k

/-- The only possible combination of integer tangents for a triangle is (1, 2, 3) -/
theorem int_tan_triangle_unique (t : IntTanTriangle) :
  (Real.tan t.α = 1 ∧ Real.tan t.β = 2 ∧ Real.tan t.γ = 3) ∨
  (Real.tan t.α = 1 ∧ Real.tan t.β = 3 ∧ Real.tan t.γ = 2) ∨
  (Real.tan t.α = 2 ∧ Real.tan t.β = 1 ∧ Real.tan t.γ = 3) ∨
  (Real.tan t.α = 2 ∧ Real.tan t.β = 3 ∧ Real.tan t.γ = 1) ∨
  (Real.tan t.α = 3 ∧ Real.tan t.β = 1 ∧ Real.tan t.γ = 2) ∨
  (Real.tan t.α = 3 ∧ Real.tan t.β = 2 ∧ Real.tan t.γ = 1) :=
by sorry

end NUMINAMATH_CALUDE_int_tan_triangle_unique_l187_18791


namespace NUMINAMATH_CALUDE_x_squared_plus_y_cubed_eq_neg_seven_l187_18737

theorem x_squared_plus_y_cubed_eq_neg_seven 
  (x y : ℝ) 
  (h : |x - 1| + (y + 2)^2 = 0) : 
  x^2 + y^3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_cubed_eq_neg_seven_l187_18737


namespace NUMINAMATH_CALUDE_no_function_satisfies_composite_condition_l187_18728

theorem no_function_satisfies_composite_condition :
  ∀ f : ℝ → ℝ, ∃ x : ℝ, f (f x) ≠ x^2 - 1996 := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_composite_condition_l187_18728


namespace NUMINAMATH_CALUDE_biased_coin_probability_l187_18759

theorem biased_coin_probability (p : ℝ) (h1 : 0 < p) (h2 : p < 1)
  (h3 : 5 * p * (1 - p)^4 = 10 * p^2 * (1 - p)^3)
  (h4 : 5 * p * (1 - p)^4 ≠ 0) :
  10 * p^3 * (1 - p)^2 = 40 / 243 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l187_18759


namespace NUMINAMATH_CALUDE_A_necessary_not_sufficient_l187_18744

-- Define proposition A
def proposition_A (a : ℝ) : Prop :=
  ∀ x, a * x^2 + 2 * a * x + 1 > 0

-- Define proposition B
def proposition_B (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Theorem stating that A is necessary but not sufficient for B
theorem A_necessary_not_sufficient :
  (∀ a, proposition_B a → proposition_A a) ∧
  ¬(∀ a, proposition_A a → proposition_B a) :=
sorry

end NUMINAMATH_CALUDE_A_necessary_not_sufficient_l187_18744


namespace NUMINAMATH_CALUDE_shifted_quadratic_sum_of_coefficients_l187_18746

-- Define the original quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- Define the shifted function
def g (x : ℝ) : ℝ := f (x + 3)

-- Theorem statement
theorem shifted_quadratic_sum_of_coefficients :
  ∃ (a b c : ℝ), (∀ x, g x = a * x^2 + b * x + c) ∧ (a + b + c = 51) := by
sorry

end NUMINAMATH_CALUDE_shifted_quadratic_sum_of_coefficients_l187_18746


namespace NUMINAMATH_CALUDE_base6_addition_puzzle_l187_18740

/-- Converts a base 6 number to base 10 -/
def base6_to_base10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 36 + tens * 6 + ones

/-- Converts a base 10 number to base 6 -/
def base10_to_base6 (n : Nat) : Nat × Nat × Nat :=
  let hundreds := n / 36
  let tens := (n % 36) / 6
  let ones := n % 6
  (hundreds, tens, ones)

theorem base6_addition_puzzle :
  ∃ (S H E : Nat),
    S ≠ 0 ∧ H ≠ 0 ∧ E ≠ 0 ∧
    S < 6 ∧ H < 6 ∧ E < 6 ∧
    S ≠ H ∧ S ≠ E ∧ H ≠ E ∧
    base6_to_base10 S H E + base6_to_base10 0 H E = base6_to_base10 S E S ∧
    S = 4 ∧ H = 1 ∧ E = 2 ∧
    base10_to_base6 (S + H + E) = (0, 1, 1) := by
  sorry

#eval base10_to_base6 7  -- Expected output: (0, 1, 1)

end NUMINAMATH_CALUDE_base6_addition_puzzle_l187_18740


namespace NUMINAMATH_CALUDE_square_diagonal_length_l187_18799

/-- The length of the diagonal of a square with side length 50√2 cm is 100 cm. -/
theorem square_diagonal_length :
  let side_length : ℝ := 50 * Real.sqrt 2
  let diagonal_length : ℝ := 100
  diagonal_length = Real.sqrt (2 * side_length ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_length_l187_18799


namespace NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l187_18771

/-- Arithmetic sequence with first term a and common difference d -/
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1 : ℤ) * d

/-- The 75th term of the arithmetic sequence starting with 3 and common difference 5 is 373 -/
theorem arithmetic_sequence_75th_term :
  arithmetic_sequence 3 5 75 = 373 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l187_18771


namespace NUMINAMATH_CALUDE_square_side_length_l187_18798

theorem square_side_length (P A : ℝ) (h1 : P = 12) (h2 : A = 9) : ∃ s : ℝ, s > 0 ∧ P = 4 * s ∧ A = s ^ 2 ∧ s = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l187_18798


namespace NUMINAMATH_CALUDE_biology_enrollment_percentage_l187_18726

theorem biology_enrollment_percentage (total_students : ℕ) (not_enrolled : ℕ) 
  (h1 : total_students = 880) (h2 : not_enrolled = 616) :
  (((total_students - not_enrolled : ℝ) / total_students) * 100 : ℝ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_biology_enrollment_percentage_l187_18726


namespace NUMINAMATH_CALUDE_zero_vector_magnitude_is_zero_l187_18736

/-- The magnitude of the zero vector in a 2D plane is 0. -/
theorem zero_vector_magnitude_is_zero :
  ∀ (v : ℝ × ℝ), v = (0, 0) → ‖v‖ = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_vector_magnitude_is_zero_l187_18736


namespace NUMINAMATH_CALUDE_students_per_class_l187_18751

theorem students_per_class 
  (total_students : ℕ) 
  (num_classrooms : ℕ) 
  (h1 : total_students = 120) 
  (h2 : num_classrooms = 24) 
  (h3 : total_students % num_classrooms = 0) : 
  total_students / num_classrooms = 5 := by
sorry

end NUMINAMATH_CALUDE_students_per_class_l187_18751


namespace NUMINAMATH_CALUDE_power_division_23_l187_18712

theorem power_division_23 : (23 : ℕ) ^ 11 / (23 : ℕ) ^ 8 = 12167 := by
  sorry

end NUMINAMATH_CALUDE_power_division_23_l187_18712


namespace NUMINAMATH_CALUDE_profit_percentage_l187_18755

theorem profit_percentage (cost_price selling_price : ℝ) 
  (h : 58 * cost_price = 50 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 16 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l187_18755


namespace NUMINAMATH_CALUDE_all_female_finalists_probability_l187_18778

-- Define the total number of participants
def total_participants : ℕ := 6

-- Define the number of female participants
def female_participants : ℕ := 4

-- Define the number of male participants
def male_participants : ℕ := 2

-- Define the number of finalists to be chosen
def finalists : ℕ := 3

-- Define the probability of selecting all female finalists
def prob_all_female_finalists : ℚ := (female_participants.choose finalists) / (total_participants.choose finalists)

-- Theorem statement
theorem all_female_finalists_probability :
  prob_all_female_finalists = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_all_female_finalists_probability_l187_18778


namespace NUMINAMATH_CALUDE_uncle_age_l187_18768

/-- Given Bud's age and the relationship to his uncle's age, calculate the uncle's age -/
theorem uncle_age (bud_age : ℕ) (h : bud_age = 8) : 
  3 * bud_age = 24 := by
  sorry

end NUMINAMATH_CALUDE_uncle_age_l187_18768


namespace NUMINAMATH_CALUDE_prob_at_least_one_woman_l187_18752

/-- The probability of selecting at least one woman when randomly choosing 3 people from a group of 5 men and 5 women is 5/6. -/
theorem prob_at_least_one_woman (n_men n_women n_select : ℕ) (h_men : n_men = 5) (h_women : n_women = 5) (h_select : n_select = 3) :
  let total := n_men + n_women
  let prob_no_women := (n_men.choose n_select : ℚ) / (total.choose n_select : ℚ)
  1 - prob_no_women = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_woman_l187_18752


namespace NUMINAMATH_CALUDE_cosine_B_in_special_triangle_l187_18795

theorem cosine_B_in_special_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π/2 →  -- acute angle A
  0 < B ∧ B < π/2 →  -- acute angle B
  0 < C ∧ C < π/2 →  -- acute angle C
  a = (2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2)) / Real.sin A →  -- side-angle relation
  b = (2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2)) / Real.sin B →  -- side-angle relation
  c = (2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2)) / Real.sin C →  -- side-angle relation
  a = Real.sqrt 7 →
  b = 3 →
  Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3 →
  Real.cos B = Real.sqrt 7 / 14 := by
sorry

end NUMINAMATH_CALUDE_cosine_B_in_special_triangle_l187_18795


namespace NUMINAMATH_CALUDE_females_only_in_orchestra_l187_18733

/-- Represents the membership data for the band and orchestra --/
structure MusicGroups where
  band_females : ℕ
  band_males : ℕ
  orchestra_females : ℕ
  orchestra_males : ℕ
  both_females : ℕ
  total_members : ℕ

/-- The theorem stating the number of females in the orchestra who are not in the band --/
theorem females_only_in_orchestra (mg : MusicGroups)
  (h1 : mg.band_females = 120)
  (h2 : mg.band_males = 100)
  (h3 : mg.orchestra_females = 100)
  (h4 : mg.orchestra_males = 120)
  (h5 : mg.both_females = 80)
  (h6 : mg.total_members = 260) :
  mg.orchestra_females - mg.both_females = 20 := by
  sorry

#check females_only_in_orchestra

end NUMINAMATH_CALUDE_females_only_in_orchestra_l187_18733


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l187_18727

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : Prop :=
  k * x^2 + 2 * x + 1 = 0

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x ∧ quadratic_equation k y

-- Theorem statement
theorem quadratic_roots_condition (k : ℝ) :
  has_two_distinct_real_roots k ↔ k < 1 ∧ k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l187_18727


namespace NUMINAMATH_CALUDE_tan_two_implies_specific_trig_ratio_l187_18707

theorem tan_two_implies_specific_trig_ratio (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin θ * Real.sin (π/2 - θ)) / (Real.sin θ^2 + Real.cos (2*θ) + Real.cos θ^2) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_tan_two_implies_specific_trig_ratio_l187_18707


namespace NUMINAMATH_CALUDE_travel_time_ngapara_to_zipra_l187_18704

/-- Proves that the time taken to travel from Ngapara to Zipra is 60 hours -/
theorem travel_time_ngapara_to_zipra (time_ngapara_zipra : ℝ) 
  (h1 : 0.8 * time_ngapara_zipra + time_ngapara_zipra = 108) : 
  time_ngapara_zipra = 60 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_ngapara_to_zipra_l187_18704


namespace NUMINAMATH_CALUDE_number_relationship_l187_18775

theorem number_relationship : 4^(3/10) < 8^(1/4) ∧ 8^(1/4) < 3^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_l187_18775


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l187_18769

/-- Given an arithmetic sequence {aₙ} with a₃ = 0 and a₁ = 4, 
    the common difference d is -2. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h1 : a 3 = 0) 
  (h2 : a 1 = 4) 
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) :
  a 2 - a 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l187_18769


namespace NUMINAMATH_CALUDE_min_value_theorem_l187_18747

theorem min_value_theorem (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_eq_one : x + y + z + w = 1) :
  (x + y + z) / (x * y * z * w) ≥ 144 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l187_18747


namespace NUMINAMATH_CALUDE_vehicle_y_speed_l187_18730

/-- Proves that the average speed of vehicle Y is 45 miles per hour given the problem conditions -/
theorem vehicle_y_speed
  (initial_distance : ℝ)
  (vehicle_x_speed : ℝ)
  (overtake_time : ℝ)
  (final_lead : ℝ)
  (h1 : initial_distance = 22)
  (h2 : vehicle_x_speed = 36)
  (h3 : overtake_time = 5)
  (h4 : final_lead = 23) :
  (initial_distance + final_lead + vehicle_x_speed * overtake_time) / overtake_time = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_vehicle_y_speed_l187_18730


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l187_18790

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 + 6*x + 8 = -(x + 2)*(x + 6) ↔ x = -2 ∨ x = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l187_18790


namespace NUMINAMATH_CALUDE_divisibility_implies_k_value_l187_18748

/-- 
If x^2 + kx - 3 is divisible by (x - 1), then k = 2.
-/
theorem divisibility_implies_k_value (k : ℤ) : 
  (∀ x : ℤ, (x - 1) ∣ (x^2 + k*x - 3)) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_k_value_l187_18748


namespace NUMINAMATH_CALUDE_quadratic_function_value_l187_18787

/-- A quadratic function with specified properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_value (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x ≤ 75) ∧ 
  (QuadraticFunction a b c (-3) = 0) ∧ 
  (QuadraticFunction a b c 3 = 0) →
  QuadraticFunction a b c 2 = 125/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l187_18787


namespace NUMINAMATH_CALUDE_divisibility_by_3804_l187_18703

theorem divisibility_by_3804 (n : ℕ+) :
  ∃ k : ℤ, (n.val^3 - n.val : ℤ) * (5^(8*n.val+4) + 3^(4*n.val+2)) = 3804 * k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_3804_l187_18703


namespace NUMINAMATH_CALUDE_investment_proof_l187_18770

/-- Represents the monthly interest rate as a decimal -/
def monthly_interest_rate : ℝ := 0.10

/-- Calculates the total amount after n months given an initial investment -/
def total_after_n_months (initial_investment : ℝ) (n : ℕ) : ℝ :=
  initial_investment * (1 + monthly_interest_rate) ^ n

/-- Theorem stating that an initial investment of $300 results in $363 after 2 months -/
theorem investment_proof :
  ∃ (initial_investment : ℝ),
    total_after_n_months initial_investment 2 = 363 ∧
    initial_investment = 300 :=
by
  sorry


end NUMINAMATH_CALUDE_investment_proof_l187_18770


namespace NUMINAMATH_CALUDE_c_neq_zero_necessary_not_sufficient_l187_18716

/-- Represents a conic section of the form ax^2 + y^2 = c -/
structure ConicSection where
  a : ℝ
  c : ℝ

/-- Determines if a conic section is an ellipse or hyperbola -/
def is_ellipse_or_hyperbola (conic : ConicSection) : Prop :=
  -- We don't define this explicitly as it's not given in the problem conditions
  sorry

/-- Theorem stating that c ≠ 0 is necessary but not sufficient for
    ax^2 + y^2 = c to represent an ellipse or hyperbola -/
theorem c_neq_zero_necessary_not_sufficient :
  (∀ conic : ConicSection, is_ellipse_or_hyperbola conic → conic.c ≠ 0) ∧
  (∃ conic : ConicSection, conic.c ≠ 0 ∧ ¬is_ellipse_or_hyperbola conic) :=
by
  sorry

end NUMINAMATH_CALUDE_c_neq_zero_necessary_not_sufficient_l187_18716


namespace NUMINAMATH_CALUDE_novel_pages_per_hour_l187_18720

-- Define the reading time in hours
def total_reading_time : ℚ := 1/6 * 24

-- Define the reading time for each type of book
def reading_time_per_type : ℚ := total_reading_time / 3

-- Define the pages read per hour for comic books and graphic novels
def comic_pages_per_hour : ℕ := 45
def graphic_pages_per_hour : ℕ := 30

-- Define the total pages read
def total_pages_read : ℕ := 128

-- Theorem to prove
theorem novel_pages_per_hour : 
  ∃ (n : ℕ), 
    (n : ℚ) * reading_time_per_type + 
    (comic_pages_per_hour : ℚ) * reading_time_per_type + 
    (graphic_pages_per_hour : ℚ) * reading_time_per_type = total_pages_read ∧ 
    n = 21 := by
  sorry

end NUMINAMATH_CALUDE_novel_pages_per_hour_l187_18720


namespace NUMINAMATH_CALUDE_ellipse_equation_l187_18732

/-- The locus of points P such that |F₁F₂| is the arithmetic mean of |PF₁| and |PF₂| -/
def EllipseLocus (F₁ F₂ : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | dist F₁ F₂ = (dist P F₁ + dist P F₂) / 2}

theorem ellipse_equation (P : ℝ × ℝ) :
  P ∈ EllipseLocus (-2, 0) (2, 0) ↔ P.1^2 / 16 + P.2^2 / 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l187_18732


namespace NUMINAMATH_CALUDE_election_votes_total_l187_18725

theorem election_votes_total (winning_percentage : ℚ) (majority : ℕ) (total_votes : ℕ) : 
  winning_percentage = 60 / 100 →
  majority = 1300 →
  winning_percentage * total_votes - (1 - winning_percentage) * total_votes = majority →
  total_votes = 6500 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_total_l187_18725


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l187_18788

theorem square_perimeter_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_area_ratio : a^2 / b^2 = 16 / 25) :
  (4 * a) / (4 * b) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l187_18788


namespace NUMINAMATH_CALUDE_cookie_count_l187_18789

theorem cookie_count (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) :
  bags * cookies_per_bag = 703 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l187_18789


namespace NUMINAMATH_CALUDE_cube_increase_theorem_l187_18714

theorem cube_increase_theorem :
  let s : ℝ := 1  -- Initial side length (can be any positive real number)
  let s' : ℝ := 1.2 * s  -- New side length after 20% increase
  let A : ℝ := 6 * s^2  -- Initial surface area
  let V : ℝ := s^3  -- Initial volume
  let A' : ℝ := 6 * s'^2  -- New surface area
  let V' : ℝ := s'^3  -- New volume
  let x : ℝ := (A' - A) / A * 100  -- Percentage increase in surface area
  let y : ℝ := (V' - V) / V * 100  -- Percentage increase in volume
  5 * (y - x) = 144 := by sorry

end NUMINAMATH_CALUDE_cube_increase_theorem_l187_18714


namespace NUMINAMATH_CALUDE_ninth_term_of_geometric_sequence_l187_18706

/-- A geometric sequence with a₃ = 16 and a₆ = 144 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ 
  (∀ n : ℕ, a (n + 1) = a n * r) ∧
  a 3 = 16 ∧ 
  a 6 = 144

theorem ninth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  a 9 = 1296 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_geometric_sequence_l187_18706


namespace NUMINAMATH_CALUDE_system_solution_l187_18754

theorem system_solution (x y : ℝ) : 
  ((x = 0 ∧ y = 0) ∨ 
   (x = 1 ∧ y = 1) ∨ 
   (x = -(5/4)^(1/5) ∧ y = (-50)^(1/5))) → 
  (4 * x^2 - 3 * y = x * y^3 ∧ 
   x^2 + x^3 * y^2 = 2 * y) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l187_18754


namespace NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_39_l187_18734

theorem right_triangle_with_hypotenuse_39 :
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  c = 39 →           -- Hypotenuse length is 39
  (a = 15 ∧ b = 36) ∨ (a = 36 ∧ b = 15) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_39_l187_18734


namespace NUMINAMATH_CALUDE_tan_condition_l187_18729

open Real

theorem tan_condition (k : ℤ) (x : ℝ) : 
  (∃ k, x = 2 * k * π + π/4) → tan x = 1 ∧ 
  ∃ x, tan x = 1 ∧ ∀ k, x ≠ 2 * k * π + π/4 :=
by sorry

end NUMINAMATH_CALUDE_tan_condition_l187_18729


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l187_18776

/-- The equation represents a circle if and only if this condition holds -/
def is_circle (a : ℝ) : Prop := 4 + 4 - 4*a > 0

/-- The condition we're examining -/
def condition (a : ℝ) : Prop := a ≤ 2

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, is_circle a → condition a) ∧
  ¬(∀ a : ℝ, condition a → is_circle a) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l187_18776


namespace NUMINAMATH_CALUDE_vector_sum_proof_l187_18710

theorem vector_sum_proof :
  let v₁ : Fin 3 → ℝ := ![3, -2, 7]
  let v₂ : Fin 3 → ℝ := ![-1, 5, -3]
  v₁ + v₂ = ![2, 3, 4] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l187_18710


namespace NUMINAMATH_CALUDE_arc_length_150_degrees_l187_18749

/-- The arc length of a circle with radius 1 cm and central angle 150° is (5π/6) cm. -/
theorem arc_length_150_degrees : 
  let radius : ℝ := 1
  let central_angle_degrees : ℝ := 150
  let central_angle_radians : ℝ := central_angle_degrees * (π / 180)
  let arc_length : ℝ := radius * central_angle_radians
  arc_length = (5 * π) / 6 := by sorry

end NUMINAMATH_CALUDE_arc_length_150_degrees_l187_18749


namespace NUMINAMATH_CALUDE_unique_c_value_l187_18762

theorem unique_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c) (h_sum : a + b + c = 11)
  (h_frac : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1) : c = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_value_l187_18762


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l187_18719

-- Define the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Define the property of being a pure imaginary number
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem complex_product_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((1 + i) * (1 + a * i)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l187_18719


namespace NUMINAMATH_CALUDE_min_area_triangle_abc_l187_18761

/-- Triangle ABC with A at origin, B at (48,18), and C with integer coordinates has minimum area 3 -/
theorem min_area_triangle_abc : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (48, 18)
  ∃ (min_area : ℝ), min_area = 3 ∧ 
    ∀ (C : ℤ × ℤ), 
      let area := (1/2) * |A.1 * (B.2 - C.2) + B.1 * C.2 + C.1 * A.2 - (B.2 * C.1 + A.1 * B.2 + C.2 * A.1)|
      area ≥ min_area :=
by
  sorry


end NUMINAMATH_CALUDE_min_area_triangle_abc_l187_18761


namespace NUMINAMATH_CALUDE_randy_biscuits_l187_18724

/-- The number of biscuits Randy is left with after receiving and losing some -/
def biscuits_left (initial : ℕ) (father_gift : ℕ) (mother_gift : ℕ) (brother_ate : ℕ) : ℕ :=
  initial + father_gift + mother_gift - brother_ate

/-- Theorem stating that Randy is left with 40 biscuits -/
theorem randy_biscuits :
  biscuits_left 32 13 15 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_randy_biscuits_l187_18724


namespace NUMINAMATH_CALUDE_quadratic_value_at_3_l187_18731

/-- A quadratic function y = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  min_value : ℝ
  min_x : ℝ
  point_x : ℝ
  point_y : ℝ

/-- Properties of the quadratic function -/
def has_properties (f : QuadraticFunction) : Prop :=
  f.min_value = -8 ∧
  f.min_x = -2 ∧
  f.point_x = 1 ∧
  f.point_y = 5 ∧
  f.point_y = f.a * f.point_x^2 + f.b * f.point_x + f.c

/-- The value of y when x = 3 -/
def y_at_3 (f : QuadraticFunction) : ℝ :=
  f.a * 3^2 + f.b * 3 + f.c

/-- The main theorem -/
theorem quadratic_value_at_3 (f : QuadraticFunction) (h : has_properties f) :
  y_at_3 f = 253/9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_value_at_3_l187_18731


namespace NUMINAMATH_CALUDE_mikes_earnings_l187_18786

/-- Calculates the total earnings from selling working video games -/
def calculate_earnings (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Proves that Mike's earnings from selling his working video games is $56 -/
theorem mikes_earnings : 
  calculate_earnings 16 8 7 = 56 := by
  sorry

end NUMINAMATH_CALUDE_mikes_earnings_l187_18786


namespace NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l187_18774

theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, x^2 + b*x + 5 ≠ -2) ↔ b ∈ Set.Ioo (-Real.sqrt 28) (Real.sqrt 28) := by
  sorry

end NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l187_18774


namespace NUMINAMATH_CALUDE_total_stairs_l187_18781

def stairs_problem (samir veronica ravi : ℕ) : Prop :=
  samir = 318 ∧
  veronica = (samir / 2 + 18) ∧
  ravi = (veronica * 3 / 2 : ℕ) ∧  -- Using integer division
  samir + veronica + ravi = 761

theorem total_stairs : ∃ samir veronica ravi : ℕ, stairs_problem samir veronica ravi :=
sorry

end NUMINAMATH_CALUDE_total_stairs_l187_18781


namespace NUMINAMATH_CALUDE_work_scaling_l187_18793

theorem work_scaling (people : ℕ) (work : ℕ) (days : ℕ) 
  (h1 : people = 3)
  (h2 : work = 3)
  (h3 : days = 3)
  (h4 : people * work = people * days) :
  (9 : ℕ) * work = 9 * days := by
  sorry

end NUMINAMATH_CALUDE_work_scaling_l187_18793


namespace NUMINAMATH_CALUDE_domino_arrangement_theorem_l187_18794

/-- Represents a domino piece -/
structure Domino :=
  (first : Nat)
  (second : Nat)
  (h1 : first ≤ 6)
  (h2 : second ≤ 6)

/-- Represents a set of dominoes -/
def DominoSet := List Domino

/-- Represents a square frame made of dominoes -/
structure Frame :=
  (dominoes : List Domino)
  (side_sum : Nat)

/-- The total number of points in a standard set of dominoes minus doubles 3, 4, 5, and 6 -/
def total_points : Nat := 132

/-- The number of frames to be formed -/
def num_frames : Nat := 3

/-- Theorem: It's possible to arrange 24 dominoes into 3 square frames with equal side sums -/
theorem domino_arrangement_theorem (dominoes : DominoSet) 
  (h1 : dominoes.length = 24)
  (h2 : (dominoes.map (λ d => d.first + d.second)).sum = total_points) :
  ∃ (frames : List Frame), 
    frames.length = num_frames ∧ 
    (∀ f ∈ frames, f.dominoes.length = 8 ∧ 
      (f.dominoes.map (λ d => d.first + d.second)).sum = total_points / num_frames) ∧
    (∀ f ∈ frames, ∀ side : List Domino, side.length = 3 → 
      (side.map (λ d => d.first + d.second)).sum = f.side_sum) := by
  sorry


end NUMINAMATH_CALUDE_domino_arrangement_theorem_l187_18794


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l187_18782

theorem triangle_inequality_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c) + b / (a + c) + c / (a + b) < 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l187_18782


namespace NUMINAMATH_CALUDE_stating_tour_days_correct_l187_18773

/-- Represents the number of days Mr. Bhaskar is on tour -/
def tour_days : ℕ := 20

/-- Total budget for the tour -/
def total_budget : ℕ := 360

/-- Number of days the tour could be extended -/
def extension_days : ℕ := 4

/-- Amount by which daily expenses must be reduced if tour is extended -/
def expense_reduction : ℕ := 3

/-- 
Theorem stating that tour_days satisfies the given conditions:
1. The total budget divided by tour_days gives the daily expense
2. If the tour is extended by extension_days, the new daily expense is 
   reduced by expense_reduction
3. The total expenditure remains the same in both scenarios
-/
theorem tour_days_correct : 
  (total_budget / tour_days) * tour_days = 
  ((total_budget / tour_days) - expense_reduction) * (tour_days + extension_days) := by
  sorry

#check tour_days_correct

end NUMINAMATH_CALUDE_stating_tour_days_correct_l187_18773


namespace NUMINAMATH_CALUDE_jorges_total_goals_l187_18722

/-- The total number of goals Jorge scored over two seasons -/
def total_goals (last_season_goals this_season_goals : ℕ) : ℕ :=
  last_season_goals + this_season_goals

/-- Theorem stating that Jorge's total goals over two seasons is 343 -/
theorem jorges_total_goals :
  total_goals 156 187 = 343 := by
  sorry

end NUMINAMATH_CALUDE_jorges_total_goals_l187_18722


namespace NUMINAMATH_CALUDE_parallelogram_point_C_l187_18780

structure Point where
  x : ℝ
  y : ℝ

def Parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = C.x - D.x) ∧ (B.y - A.y = C.y - D.y) ∧
  (D.x - A.x = C.x - B.x) ∧ (D.y - A.y = C.y - B.y)

def InFirstQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y > 0

theorem parallelogram_point_C : 
  ∀ (A B C D : Point),
    Parallelogram A B C D →
    InFirstQuadrant A →
    InFirstQuadrant B →
    InFirstQuadrant C →
    InFirstQuadrant D →
    A.x = 2 ∧ A.y = 3 →
    B.x = 7 ∧ B.y = 3 →
    D.x = 3 ∧ D.y = 7 →
    C.x = 8 ∧ C.y = 7 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_point_C_l187_18780


namespace NUMINAMATH_CALUDE_min_value_and_inequality_range_l187_18743

theorem min_value_and_inequality_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 3) :
  (∃ (min : ℝ), min = 6 ∧ ∀ x y, x > 0 → y > 0 → x * y = 3 → x + 3 * y ≥ min) ∧
  (∀ m : ℝ, (∀ x y, x > 0 → y > 0 → x * y = 3 → m^2 - (x + 3 * y) * m + 5 ≤ 0) → 1 ≤ m ∧ m ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_range_l187_18743


namespace NUMINAMATH_CALUDE_barbecue_sauce_ketchup_amount_l187_18772

theorem barbecue_sauce_ketchup_amount :
  let total_sauce := k + 1 + 1
  let burger_sauce := (1 : ℚ) / 4
  let sandwich_sauce := (1 : ℚ) / 6
  let num_burgers := 8
  let num_sandwiches := 18
  ∀ k : ℚ,
  (num_burgers * burger_sauce + num_sandwiches * sandwich_sauce = total_sauce) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_barbecue_sauce_ketchup_amount_l187_18772


namespace NUMINAMATH_CALUDE_new_manufacturing_cost_l187_18763

/-- Given a constant selling price, an initial manufacturing cost, and profit percentages,
    calculate the new manufacturing cost after a change in profit percentage. -/
theorem new_manufacturing_cost
  (P : ℝ)  -- Selling price
  (initial_cost : ℝ)  -- Initial manufacturing cost
  (initial_profit_percent : ℝ)  -- Initial profit as a percentage of selling price
  (new_profit_percent : ℝ)  -- New profit as a percentage of selling price
  (h1 : initial_cost = 80)  -- Initial cost is $80
  (h2 : initial_profit_percent = 0.20)  -- Initial profit is 20%
  (h3 : new_profit_percent = 0.50)  -- New profit is 50%
  (h4 : P - initial_cost = initial_profit_percent * P)  -- Initial profit equation
  : P - new_profit_percent * P = 50 := by
  sorry


end NUMINAMATH_CALUDE_new_manufacturing_cost_l187_18763


namespace NUMINAMATH_CALUDE_sequence_decreasing_two_equal_max_terms_l187_18708

-- Define the sequence aₙ
def a (k : ℝ) (n : ℕ) : ℝ := n * k^n

-- Proposition ②
theorem sequence_decreasing (k : ℝ) (h1 : 0 < k) (h2 : k < 1/2) :
  ∀ n : ℕ, n > 0 → a k (n + 1) < a k n :=
sorry

-- Proposition ④
theorem two_equal_max_terms (k : ℝ) (h : ∃ m : ℕ, m > 0 ∧ k / (1 - k) = m) :
  ∃ n : ℕ, n > 0 ∧ a k n = a k (n + 1) ∧ ∀ m : ℕ, m > 0 → a k m ≤ a k n :=
sorry

end NUMINAMATH_CALUDE_sequence_decreasing_two_equal_max_terms_l187_18708


namespace NUMINAMATH_CALUDE_x_positive_iff_sum_geq_two_l187_18741

theorem x_positive_iff_sum_geq_two (x : ℝ) : x > 0 ↔ x + 1/x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_x_positive_iff_sum_geq_two_l187_18741


namespace NUMINAMATH_CALUDE_area_of_triangle_WRX_l187_18745

-- Define the points
variable (W X Y Z P Q R : ℝ × ℝ)

-- Define the conditions
def is_rectangle (A B C D : ℝ × ℝ) : Prop := sorry
def on_line (P A B : ℝ × ℝ) : Prop := sorry
def distance (A B : ℝ × ℝ) : ℝ := sorry
def intersect (A B C D E : ℝ × ℝ) : Prop := sorry
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_WRX 
  (h1 : is_rectangle W X Y Z)
  (h2 : distance W Z = 7)
  (h3 : distance X Y = 4)
  (h4 : on_line P Y Z)
  (h5 : on_line Q Y Z)
  (h6 : distance Y P = 2)
  (h7 : distance Q Z = 3)
  (h8 : intersect W P X Q R) :
  area_triangle W R X = 98/5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_WRX_l187_18745


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l187_18723

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 72) (h2 : Nat.gcd a c = 240) :
  (∃ (x y z : ℕ+), x = a ∧ y = b ∧ z = c ∧ Nat.gcd y z = 24) ∧
  (∀ (p q : ℕ+), Nat.gcd p q < 24 → ¬(∃ (r : ℕ+), Nat.gcd r p = 72 ∧ Nat.gcd r q = 240)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l187_18723


namespace NUMINAMATH_CALUDE_num_numerators_for_T_l187_18792

/-- The set of all rational numbers with repeating decimal expansion 0.efghefgh... -/
def T : Set ℚ :=
  {r : ℚ | 0 < r ∧ r < 1 ∧ ∃ (e f g h : ℕ), r = (e * 1000 + f * 100 + g * 10 + h) / 9999}

/-- The number of different numerators required to write elements of T in lowest terms -/
def num_numerators : ℕ := Nat.totient 9999

theorem num_numerators_for_T : num_numerators = 6000 := by
  sorry

end NUMINAMATH_CALUDE_num_numerators_for_T_l187_18792


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l187_18711

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sin x}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l187_18711


namespace NUMINAMATH_CALUDE_black_burger_cost_l187_18760

theorem black_burger_cost (salmon_cost chicken_cost total_bill : ℝ) 
  (h1 : salmon_cost = 40)
  (h2 : chicken_cost = 25)
  (h3 : total_bill = 92) : 
  ∃ (burger_cost : ℝ), 
    burger_cost = 15 ∧ 
    total_bill = (salmon_cost + burger_cost + chicken_cost) * 1.15 := by
  sorry

end NUMINAMATH_CALUDE_black_burger_cost_l187_18760


namespace NUMINAMATH_CALUDE_josh_has_eight_riddles_l187_18715

/-- The number of riddles each person has -/
structure Riddles where
  taso : ℕ
  ivory : ℕ
  josh : ℕ

/-- Given conditions about the riddles -/
def riddle_conditions (r : Riddles) : Prop :=
  r.taso = 24 ∧
  r.taso = 2 * r.ivory ∧
  r.ivory = r.josh + 4

/-- Theorem stating that Josh has 8 riddles -/
theorem josh_has_eight_riddles (r : Riddles) 
  (h : riddle_conditions r) : r.josh = 8 := by
  sorry

end NUMINAMATH_CALUDE_josh_has_eight_riddles_l187_18715


namespace NUMINAMATH_CALUDE_floor_equality_iff_in_interval_l187_18739

theorem floor_equality_iff_in_interval (x : ℝ) :
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋ ↔ x ∈ Set.Icc (5/3) (7/3) := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_iff_in_interval_l187_18739


namespace NUMINAMATH_CALUDE_largest_integer_with_conditions_l187_18785

/-- A function that returns the digits of an integer -/
def digits (n : ℕ) : List ℕ := sorry

/-- A function that checks if each digit is twice the previous one -/
def doubling_digits (l : List ℕ) : Prop := sorry

/-- A function that calculates the sum of squares of a list of digits -/
def sum_of_squares (l : List ℕ) : ℕ := sorry

/-- A function that calculates the product of a list of digits -/
def product_of_digits (l : List ℕ) : ℕ := sorry

theorem largest_integer_with_conditions (n : ℕ) :
  (∀ m : ℕ, m > n → ¬(sum_of_squares (digits m) = 65 ∧ doubling_digits (digits m))) →
  sum_of_squares (digits n) = 65 →
  doubling_digits (digits n) →
  product_of_digits (digits n) = 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_conditions_l187_18785


namespace NUMINAMATH_CALUDE_log_x3y2_equals_2_l187_18766

theorem log_x3y2_equals_2 
  (x y : ℝ) 
  (h1 : Real.log (x^2 * y^5) = 2) 
  (h2 : Real.log (x^3 * y^2) = 2) : 
  Real.log (x^3 * y^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_log_x3y2_equals_2_l187_18766


namespace NUMINAMATH_CALUDE_book_cost_proof_l187_18753

-- Define the cost of one book
def p : ℝ := 1.76

-- State the theorem
theorem book_cost_proof :
  14 * p < 25 ∧ 16 * p > 28 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_proof_l187_18753


namespace NUMINAMATH_CALUDE_plane_q_satisfies_conditions_l187_18700

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Represents a point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if a plane contains a line defined by the intersection of two other planes -/
def containsIntersectionLine (p : Plane) (p1 p2 : Plane) : Prop := sorry

/-- Calculates the distance from a plane to a point -/
def distanceToPoint (p : Plane) (pt : Point) : ℝ := sorry

/-- Checks if two planes are different -/
def areDifferentPlanes (p1 p2 : Plane) : Prop := sorry

/-- Calculates the greatest common divisor of four integers -/
def gcd4 (a b c d : ℤ) : ℕ := sorry

theorem plane_q_satisfies_conditions : 
  let p1 : Plane := { a := 2, b := -1, c := 3, d := -4 }
  let p2 : Plane := { a := 3, b := 2, c := -1, d := -6 }
  let q : Plane := { a := 0, b := -7, c := 11, d := -6 }
  let pt : Point := { x := 2, y := -2, z := 1 }
  containsIntersectionLine q p1 p2 ∧ 
  areDifferentPlanes q p1 ∧
  areDifferentPlanes q p2 ∧
  distanceToPoint q pt = 3 / Real.sqrt 5 ∧
  q.a > 0 ∧
  gcd4 (Int.natAbs q.a) (Int.natAbs q.b) (Int.natAbs q.c) (Int.natAbs q.d) = 1 := by
  sorry

end NUMINAMATH_CALUDE_plane_q_satisfies_conditions_l187_18700


namespace NUMINAMATH_CALUDE_vector_c_coordinates_l187_18750

def a : Fin 3 → ℝ := ![0, 1, -1]
def b : Fin 3 → ℝ := ![1, 2, 3]
def c : Fin 3 → ℝ := λ i => 3 * a i - b i

theorem vector_c_coordinates :
  c = ![-1, 1, -6] := by sorry

end NUMINAMATH_CALUDE_vector_c_coordinates_l187_18750


namespace NUMINAMATH_CALUDE_min_ab_value_l187_18764

-- Define the triangle ABC
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Define the condition 2c * cos B = 2a + b
def condition1 (a b c : ℝ) (B : ℝ) : Prop :=
  2 * c * Real.cos B = 2 * a + b

-- Define the area condition S = (√3/2) * c
def condition2 (c : ℝ) (S : ℝ) : Prop :=
  S = (Real.sqrt 3 / 2) * c

-- Theorem statement
theorem min_ab_value (a b c : ℝ) (B : ℝ) (S : ℝ) :
  triangle a b c →
  condition1 a b c B →
  condition2 c S →
  a * b ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_min_ab_value_l187_18764


namespace NUMINAMATH_CALUDE_solve_equation_l187_18702

theorem solve_equation (x : ℝ) :
  Real.sqrt ((3 / x) + 3 * x) = 3 →
  x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l187_18702


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l187_18709

/-- 
Given a repeating decimal 0.6̄13̄ (where 13 repeats infinitely after 6),
prove that it is equal to the fraction 362/495.
-/
theorem repeating_decimal_to_fraction : 
  (6/10 : ℚ) + (13/99 : ℚ) = 362/495 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l187_18709


namespace NUMINAMATH_CALUDE_set_equality_l187_18738

theorem set_equality (A B X : Set α) 
  (h1 : A ∩ X = B ∩ X)
  (h2 : A ∩ X = A ∩ B)
  (h3 : A ∪ B ∪ X = A ∪ B) : 
  X = A ∩ B := by
sorry

end NUMINAMATH_CALUDE_set_equality_l187_18738


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l187_18721

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional inning -/
def newAverage (stats : BatsmanStats) (newInningScore : ℕ) : ℚ :=
  (stats.totalRuns + newInningScore) / (stats.innings + 1)

/-- Theorem: If a batsman's average increases by 5 after scoring 110 in the 11th inning, 
    then the new average is 60 -/
theorem batsman_average_theorem (stats : BatsmanStats) 
  (h1 : stats.innings = 10)
  (h2 : newAverage stats 110 = stats.average + 5) :
  newAverage stats 110 = 60 := by
  sorry

#check batsman_average_theorem

end NUMINAMATH_CALUDE_batsman_average_theorem_l187_18721


namespace NUMINAMATH_CALUDE_data_analytics_course_hours_l187_18758

/-- Represents the total hours spent on a data analytics course -/
def course_total_hours (weeks : ℕ) (weekly_class_hours : ℕ) (weekly_homework_hours : ℕ) 
  (lab_sessions : ℕ) (lab_session_hours : ℕ) (project_hours : List ℕ) : ℕ :=
  weeks * (weekly_class_hours + weekly_homework_hours) + 
  lab_sessions * lab_session_hours + 
  project_hours.sum

/-- Theorem stating the total hours spent on the specific data analytics course -/
theorem data_analytics_course_hours : 
  course_total_hours 24 10 4 8 6 [10, 14, 18] = 426 := by
  sorry

end NUMINAMATH_CALUDE_data_analytics_course_hours_l187_18758


namespace NUMINAMATH_CALUDE_lemonade_water_requirement_l187_18797

/-- The amount of water required for lemonade recipe -/
def water_required (water_parts : ℚ) (lemon_juice_parts : ℚ) (total_gallons : ℚ) (quarts_per_gallon : ℚ) (cups_per_quart : ℚ) : ℚ :=
  (water_parts / (water_parts + lemon_juice_parts)) * total_gallons * quarts_per_gallon * cups_per_quart

/-- Theorem stating the required amount of water for the lemonade recipe -/
theorem lemonade_water_requirement : 
  water_required 5 2 (3/2) 4 4 = 120/7 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_water_requirement_l187_18797


namespace NUMINAMATH_CALUDE_min_lines_same_quadrant_l187_18767

/-- A line in a Cartesian coordinate system --/
structure Line where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0

/-- A family of lines in a Cartesian coordinate system --/
def LineFamily := Set Line

/-- The minimum number of lines needed to guarantee at least two lines in the same quadrant --/
def minLinesForSameQuadrant (family : LineFamily) : ℕ := 7

/-- Theorem stating that 7 is the minimum number of lines needed --/
theorem min_lines_same_quadrant (family : LineFamily) :
  minLinesForSameQuadrant family = 7 :=
sorry

end NUMINAMATH_CALUDE_min_lines_same_quadrant_l187_18767


namespace NUMINAMATH_CALUDE_distance_to_pole_l187_18796

def polar_distance (ρ : ℝ) (θ : ℝ) : ℝ := ρ

theorem distance_to_pole (A : ℝ × ℝ) (h : A = (3, -4)) :
  polar_distance A.1 A.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_pole_l187_18796


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l187_18777

theorem polynomial_evaluation : 
  103^5 - 5 * 103^4 + 10 * 103^3 - 10 * 103^2 + 5 * 103 - 1 = 11036846832 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l187_18777


namespace NUMINAMATH_CALUDE_elliptic_curve_solutions_l187_18756

theorem elliptic_curve_solutions (p : ℕ) (hp : Nat.Prime p) :
  (∃ (S : Finset (Fin p × Fin p)),
    S.card = p ∧
    ∀ (x y : Fin p), (x, y) ∈ S ↔ y^2 ≡ x^3 + 4*x [ZMOD p]) ↔
  p = 2 ∨ p ≡ 3 [MOD 4] :=
sorry

end NUMINAMATH_CALUDE_elliptic_curve_solutions_l187_18756


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l187_18705

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 3

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements in the arrangement -/
def total_elements : ℕ := num_ones + num_zeros

/-- The probability that two zeros are not adjacent when randomly arranged with three ones -/
def prob_zeros_not_adjacent : ℚ := 3/5

theorem zeros_not_adjacent_probability :
  prob_zeros_not_adjacent = 3/5 := by sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l187_18705


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l187_18779

def C : Set Nat := {65, 67, 68, 71, 74}

def has_smallest_prime_factor (n : Nat) (s : Set Nat) : Prop :=
  n ∈ s ∧ ∀ m ∈ s, ∀ p q : Nat, Prime p → Prime q → p ∣ n → q ∣ m → p ≤ q

theorem smallest_prime_factor_in_C :
  has_smallest_prime_factor 68 C := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l187_18779


namespace NUMINAMATH_CALUDE_allocation_ways_l187_18765

theorem allocation_ways (n : ℕ) (k : ℕ) (h : n = 8 ∧ k = 4) : k^n = 65536 := by
  sorry

end NUMINAMATH_CALUDE_allocation_ways_l187_18765


namespace NUMINAMATH_CALUDE_product_selection_theorem_l187_18757

/-- Represents the outcome of selecting two items from a batch of products -/
inductive Outcome
  | TwoGenuine
  | OneGenuineOneDefective
  | TwoDefective

/-- Represents a batch of products -/
structure Batch where
  genuine : ℕ
  defective : ℕ
  h_genuine : genuine > 2
  h_defective : defective > 2

/-- Probability of an outcome given a batch -/
def prob (b : Batch) (o : Outcome) : ℝ := sorry

/-- Event of having exactly one defective product -/
def exactly_one_defective (o : Outcome) : Prop :=
  o = Outcome.OneGenuineOneDefective

/-- Event of having exactly two defective products -/
def exactly_two_defective (o : Outcome) : Prop :=
  o = Outcome.TwoDefective

/-- Event of having at least one defective product -/
def at_least_one_defective (o : Outcome) : Prop :=
  o = Outcome.OneGenuineOneDefective ∨ o = Outcome.TwoDefective

/-- Event of having all genuine products -/
def all_genuine (o : Outcome) : Prop :=
  o = Outcome.TwoGenuine

theorem product_selection_theorem (b : Batch) :
  -- Statement ②: Exactly one defective and exactly two defective are mutually exclusive
  (∀ o : Outcome, ¬(exactly_one_defective o ∧ exactly_two_defective o)) ∧
  -- Statement ④: At least one defective and all genuine are mutually exclusive and complementary
  (∀ o : Outcome, ¬(at_least_one_defective o ∧ all_genuine o)) ∧
  (∀ o : Outcome, at_least_one_defective o ∨ all_genuine o) ∧
  -- Statements ① and ③ are incorrect (we don't need to prove them, just state that they're not included)
  True := by sorry

end NUMINAMATH_CALUDE_product_selection_theorem_l187_18757


namespace NUMINAMATH_CALUDE_final_collection_is_55_l187_18784

def museum_donations (initial_collection : ℕ) : ℕ :=
  let guggenheim_donation := 51
  let metropolitan_donation := 2 * guggenheim_donation
  let damaged_sets := 20
  let after_damage := initial_collection - guggenheim_donation - metropolitan_donation - damaged_sets
  let louvre_donation := after_damage / 2
  let after_louvre := after_damage - louvre_donation
  let british_donation := (2 * after_louvre) / 3
  after_louvre - british_donation

theorem final_collection_is_55 :
  museum_donations 500 = 55 := by
  sorry

end NUMINAMATH_CALUDE_final_collection_is_55_l187_18784


namespace NUMINAMATH_CALUDE_M_congruent_544_mod_1000_l187_18735

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def M : ℕ :=
  let total_blue := 20
  let total_green := 15
  let total_slots := total_blue + 1
  let ways_to_arrange_greens := Nat.choose total_slots total_green
  let ways_to_divide_poles := total_slots
  let arrangements_with_empty_pole := Nat.choose total_blue total_green
  ways_to_divide_poles * ways_to_arrange_greens - 2 * arrangements_with_empty_pole

/-- The theorem stating that M is congruent to 544 modulo 1000 -/
theorem M_congruent_544_mod_1000 : M ≡ 544 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_M_congruent_544_mod_1000_l187_18735
