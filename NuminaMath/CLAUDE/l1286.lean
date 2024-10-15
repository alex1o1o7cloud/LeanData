import Mathlib

namespace NUMINAMATH_CALUDE_division_by_fraction_not_always_larger_l1286_128640

theorem division_by_fraction_not_always_larger : ∃ (a b c : ℚ), b ≠ 0 ∧ c ≠ 0 ∧ (a / (b / c)) ≤ a := by
  sorry

end NUMINAMATH_CALUDE_division_by_fraction_not_always_larger_l1286_128640


namespace NUMINAMATH_CALUDE_set_difference_equals_singleton_l1286_128637

-- Define the set I
def I : Set ℕ := {x | 1 < x ∧ x < 5}

-- Define the set A
def A : Set ℕ := {2, 3}

-- Theorem statement
theorem set_difference_equals_singleton :
  I \ A = {4} := by sorry

end NUMINAMATH_CALUDE_set_difference_equals_singleton_l1286_128637


namespace NUMINAMATH_CALUDE_equations_represent_problem_l1286_128616

/-- Represents the money held by person A -/
def money_A : ℝ := sorry

/-- Represents the money held by person B -/
def money_B : ℝ := sorry

/-- The system of equations representing the problem -/
def problem_equations (x y : ℝ) : Prop :=
  (x + (1/2) * y = 50) ∧ (y + (2/3) * x = 50)

/-- Theorem stating that the system of equations correctly represents the problem -/
theorem equations_represent_problem :
  problem_equations money_A money_B ↔
  ((money_A + (1/2) * money_B = 50) ∧
   (money_B + (2/3) * money_A = 50)) :=
sorry

end NUMINAMATH_CALUDE_equations_represent_problem_l1286_128616


namespace NUMINAMATH_CALUDE_friends_drawing_cards_l1286_128624

theorem friends_drawing_cards (n : ℕ) (h : n = 3) :
  let total_outcomes := n.factorial
  let favorable_outcomes := (n - 1).factorial
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_friends_drawing_cards_l1286_128624


namespace NUMINAMATH_CALUDE_latia_hourly_wage_l1286_128609

/-- The cost of the TV in dollars -/
def tv_cost : ℝ := 1700

/-- The number of hours Latia works per week -/
def weekly_hours : ℝ := 30

/-- The additional hours Latia needs to work to afford the TV -/
def additional_hours : ℝ := 50

/-- The number of weeks in a month -/
def weeks_per_month : ℝ := 4

/-- Latia's hourly wage in dollars -/
def hourly_wage : ℝ := 10

theorem latia_hourly_wage :
  tv_cost = (weekly_hours * weeks_per_month + additional_hours) * hourly_wage :=
by sorry

end NUMINAMATH_CALUDE_latia_hourly_wage_l1286_128609


namespace NUMINAMATH_CALUDE_consecutive_even_integers_l1286_128628

theorem consecutive_even_integers (n : ℤ) : 
  (∃ (a b c : ℤ), 
    (a = n - 2 ∧ b = n ∧ c = n + 2) ∧  -- consecutive even integers
    (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0) ∧  -- all are even
    (a + c = 128))  -- sum of first and third is 128
  → n = 64 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_l1286_128628


namespace NUMINAMATH_CALUDE_min_value_cos_sin_l1286_128647

theorem min_value_cos_sin (θ : Real) (h : θ ∈ Set.Icc (-π/12) (π/12)) :
  ∃ (m : Real), m = (Real.sqrt 3 - 1) / 2 ∧ 
    ∀ x ∈ Set.Icc (-π/12) (π/12), 
      Real.cos (x + π/4) + Real.sin (2*x) ≥ m ∧
      ∃ y ∈ Set.Icc (-π/12) (π/12), Real.cos (y + π/4) + Real.sin (2*y) = m :=
by sorry

end NUMINAMATH_CALUDE_min_value_cos_sin_l1286_128647


namespace NUMINAMATH_CALUDE_factorization_of_360_l1286_128660

theorem factorization_of_360 : ∃ (p₁ p₂ p₃ : Nat) (e₁ e₂ e₃ : Nat),
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
  360 = p₁^e₁ * p₂^e₂ * p₃^e₃ ∧
  (∀ q : Nat, Prime q → q ∣ 360 → (q = p₁ ∨ q = p₂ ∨ q = p₃)) ∧
  (e₁ ≤ 3 ∧ e₂ ≤ 3 ∧ e₃ ≤ 3) ∧
  (e₁ = 3 ∨ e₂ = 3 ∨ e₃ = 3) :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_360_l1286_128660


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_length_l1286_128681

theorem right_triangle_shorter_leg_length 
  (a : ℝ)  -- length of the shorter leg
  (h1 : a > 0)  -- ensure positive length
  (h2 : (a^2 + (2*a)^2)^(1/2) = a * 5^(1/2))  -- Pythagorean theorem
  (h3 : 12 = (1/2) * a * 5^(1/2))  -- median to hypotenuse formula
  : a = 24 * 5^(1/2) / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_length_l1286_128681


namespace NUMINAMATH_CALUDE_jane_change_l1286_128698

-- Define the cost of the apple
def apple_cost : ℚ := 75/100

-- Define the amount Jane pays
def amount_paid : ℚ := 5

-- Define the change function
def change (cost paid : ℚ) : ℚ := paid - cost

-- Theorem statement
theorem jane_change : change apple_cost amount_paid = 425/100 := by
  sorry

end NUMINAMATH_CALUDE_jane_change_l1286_128698


namespace NUMINAMATH_CALUDE_distinct_determinants_count_l1286_128604

-- Define a type for third-order determinants
def ThirdOrderDeterminant := Matrix (Fin 3) (Fin 3) ℝ

-- Define a function to calculate the number of distinct determinants
def distinctDeterminants (n : ℕ) : ℕ :=
  if n = 9 then Nat.factorial 9 / 36 else 0

theorem distinct_determinants_count :
  distinctDeterminants 9 = 10080 := by
  sorry

#eval distinctDeterminants 9

end NUMINAMATH_CALUDE_distinct_determinants_count_l1286_128604


namespace NUMINAMATH_CALUDE_like_terms_exponent_product_l1286_128664

theorem like_terms_exponent_product (a b : ℝ) (m n : ℤ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ 3 * a^m * b^2 = k * (-a^2 * b^(n+3))) → m * n = -2 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_product_l1286_128664


namespace NUMINAMATH_CALUDE_reporters_covering_local_politics_l1286_128632

theorem reporters_covering_local_politics
  (percent_not_covering_local : Real)
  (percent_not_covering_politics : Real)
  (h1 : percent_not_covering_local = 0.3)
  (h2 : percent_not_covering_politics = 0.6) :
  (1 - percent_not_covering_politics) * (1 - percent_not_covering_local) = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_reporters_covering_local_politics_l1286_128632


namespace NUMINAMATH_CALUDE_next_shared_meeting_proof_l1286_128672

/-- The number of days between drama club meetings -/
def drama_interval : ℕ := 3

/-- The number of days between choir meetings -/
def choir_interval : ℕ := 5

/-- The number of days until both groups meet again -/
def next_shared_meeting : ℕ := 30

theorem next_shared_meeting_proof :
  ∃ (n : ℕ), n > 0 ∧ n * drama_interval = next_shared_meeting ∧ n * choir_interval = next_shared_meeting :=
sorry

end NUMINAMATH_CALUDE_next_shared_meeting_proof_l1286_128672


namespace NUMINAMATH_CALUDE_cistern_leak_emptying_time_l1286_128690

/-- Given a cistern that normally fills in 8 hours, but takes 10 hours to fill with a leak,
    prove that it takes 40 hours for a full cistern to empty due to the leak. -/
theorem cistern_leak_emptying_time (normal_fill_time leak_fill_time : ℝ) 
    (h1 : normal_fill_time = 8)
    (h2 : leak_fill_time = 10) : 
  let normal_fill_rate := 1 / normal_fill_time
  let leak_rate := normal_fill_rate - (1 / leak_fill_time)
  (1 / leak_rate) = 40 := by sorry

end NUMINAMATH_CALUDE_cistern_leak_emptying_time_l1286_128690


namespace NUMINAMATH_CALUDE_probability_in_range_l1286_128606

/-- 
Given a random variable ξ with probability distribution:
P(ξ=k) = 1/(2^(k-1)) for k = 2, 3, ..., n
P(ξ=1) = a
Prove that P(2 < ξ ≤ 5) = 7/16
-/
theorem probability_in_range (n : ℕ) (a : ℝ) (ξ : ℕ → ℝ) 
  (h1 : ∀ k ∈ Finset.range (n - 1) \ {0}, ξ (k + 2) = (1 : ℝ) / 2^k)
  (h2 : ξ 1 = a) :
  (ξ 3 + ξ 4 + ξ 5) = 7/16 := by
sorry

end NUMINAMATH_CALUDE_probability_in_range_l1286_128606


namespace NUMINAMATH_CALUDE_landscape_length_l1286_128686

theorem landscape_length (breadth : ℝ) 
  (length_eq : length = 8 * breadth)
  (playground_area : ℝ)
  (playground_eq : playground_area = 1200)
  (playground_ratio : playground_area = (1/6) * (length * breadth)) : 
  length = 240 := by
  sorry

end NUMINAMATH_CALUDE_landscape_length_l1286_128686


namespace NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_3_and_6_l1286_128657

theorem smallest_three_digit_divisible_by_3_and_6 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 3 = 0 ∧ n % 6 = 0 → n ≥ 102 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_3_and_6_l1286_128657


namespace NUMINAMATH_CALUDE_all_error_types_cause_random_errors_at_least_three_random_error_causes_l1286_128669

-- Define the types of errors
inductive ErrorType
  | ApproximationError
  | OmittedVariableError
  | ObservationError

-- Define a predicate for causes of random errors
def is_random_error_cause (error_type : ErrorType) : Prop :=
  match error_type with
  | ErrorType.ApproximationError => true
  | ErrorType.OmittedVariableError => true
  | ErrorType.ObservationError => true

-- Theorem stating that all three error types are causes of random errors
theorem all_error_types_cause_random_errors :
  (∀ (error_type : ErrorType), is_random_error_cause error_type) :=
by
  sorry

-- Theorem stating that there are at least three distinct causes of random errors
theorem at_least_three_random_error_causes :
  ∃ (e1 e2 e3 : ErrorType),
    e1 ≠ e2 ∧ e1 ≠ e3 ∧ e2 ≠ e3 ∧
    is_random_error_cause e1 ∧
    is_random_error_cause e2 ∧
    is_random_error_cause e3 :=
by
  sorry

end NUMINAMATH_CALUDE_all_error_types_cause_random_errors_at_least_three_random_error_causes_l1286_128669


namespace NUMINAMATH_CALUDE_triple_hash_48_l1286_128653

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.75 * N + 2

-- State the theorem
theorem triple_hash_48 : hash (hash (hash 48)) = 24.875 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_48_l1286_128653


namespace NUMINAMATH_CALUDE_area_of_DEFGHT_l1286_128623

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a square -/
structure Square :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Function to calculate the area of a shape formed by DEFGHT -/
def areaOfDEFGHT (ABC : Triangle) (ABDE : Square) (CAFG : Square) (BCHT : Triangle) : ℝ :=
  sorry

/-- Theorem stating the area of shape DEFGHT -/
theorem area_of_DEFGHT :
  ∀ (ABC : Triangle) (ABDE : Square) (CAFG : Square) (BCHT : Triangle),
  (ABC.A.x - ABC.B.x)^2 + (ABC.A.y - ABC.B.y)^2 = 4 ∧  -- Side length of ABC is 2
  (ABC.B.x - ABC.C.x)^2 + (ABC.B.y - ABC.C.y)^2 = 4 ∧
  (ABC.C.x - ABC.A.x)^2 + (ABC.C.y - ABC.A.y)^2 = 4 ∧
  (ABDE.A = ABC.A ∧ ABDE.B = ABC.B) ∧  -- ABDE is a square outside ABC
  (CAFG.C = ABC.A ∧ CAFG.A = ABC.C) ∧  -- CAFG is a square outside ABC
  (BCHT.B = ABC.B ∧ BCHT.C = ABC.C) ∧  -- BCHT is an equilateral triangle outside ABC
  (BCHT.A.x - BCHT.B.x)^2 + (BCHT.A.y - BCHT.B.y)^2 = 4 ∧  -- BCHT is equilateral with side length 2
  (BCHT.B.x - BCHT.C.x)^2 + (BCHT.B.y - BCHT.C.y)^2 = 4 ∧
  (BCHT.C.x - BCHT.A.x)^2 + (BCHT.C.y - BCHT.A.y)^2 = 4 →
  areaOfDEFGHT ABC ABDE CAFG BCHT = 3 * Real.sqrt 3 - 2 :=
by
  sorry

end NUMINAMATH_CALUDE_area_of_DEFGHT_l1286_128623


namespace NUMINAMATH_CALUDE_jerry_age_l1286_128680

theorem jerry_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 16 → 
  mickey_age = 2 * jerry_age - 6 → 
  jerry_age = 11 := by
sorry

end NUMINAMATH_CALUDE_jerry_age_l1286_128680


namespace NUMINAMATH_CALUDE_bacteria_growth_l1286_128631

/-- Calculates the bacteria population after a given time interval -/
def bacteria_population (initial_population : ℕ) (doubling_time : ℕ) (total_time : ℕ) : ℕ :=
  initial_population * 2 ^ (total_time / doubling_time)

/-- Theorem: Given 20 initial bacteria that double every 3 minutes, 
    the population after 15 minutes is 640 -/
theorem bacteria_growth : bacteria_population 20 3 15 = 640 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l1286_128631


namespace NUMINAMATH_CALUDE_charity_fundraising_l1286_128674

theorem charity_fundraising (total_amount : ℕ) (num_people : ℕ) (amount_per_person : ℕ) :
  total_amount = 1500 →
  num_people = 6 →
  amount_per_person * num_people = total_amount →
  amount_per_person = 250 := by
  sorry

end NUMINAMATH_CALUDE_charity_fundraising_l1286_128674


namespace NUMINAMATH_CALUDE_relationship_2x_3sinx_l1286_128692

theorem relationship_2x_3sinx :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧
  (∀ x : ℝ, 0 < x → x < θ → 2 * x < 3 * Real.sin x) ∧
  (2 * θ = 3 * Real.sin θ) ∧
  (∀ x : ℝ, θ < x → x < π / 2 → 2 * x > 3 * Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_relationship_2x_3sinx_l1286_128692


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1286_128642

theorem arithmetic_calculation : -8 * 4 - (-6 * -3) + (-10 * -5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1286_128642


namespace NUMINAMATH_CALUDE_inequality_proof_l1286_128601

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 2 * c > a + b) : 
  c - Real.sqrt (c^2 - a*b) < a ∧ a < c + Real.sqrt (c^2 - a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1286_128601


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1286_128634

/-- The length of the longer diagonal of a rhombus given its side length and shorter diagonal -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (h1 : side = 65) (h2 : shorter_diagonal = 56) :
  ∃ longer_diagonal : ℝ, longer_diagonal = 2 * Real.sqrt 3441 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1286_128634


namespace NUMINAMATH_CALUDE_cricket_team_size_l1286_128643

/-- Represents a cricket team with given age properties -/
structure CricketTeam where
  n : ℕ  -- number of team members
  captain_age : ℕ
  wicket_keeper_age : ℕ
  team_avg_age : ℝ
  remaining_avg_age : ℝ

/-- The cricket team satisfies the given conditions -/
def valid_cricket_team (team : CricketTeam) : Prop :=
  team.captain_age = 26 ∧
  team.wicket_keeper_age = team.captain_age + 3 ∧
  team.team_avg_age = 23 ∧
  team.remaining_avg_age = team.team_avg_age - 1 ∧
  (team.n : ℝ) * team.team_avg_age = 
    (team.n - 2 : ℝ) * team.remaining_avg_age + team.captain_age + team.wicket_keeper_age

theorem cricket_team_size (team : CricketTeam) :
  valid_cricket_team team → team.n = 11 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l1286_128643


namespace NUMINAMATH_CALUDE_triangle_area_l1286_128697

theorem triangle_area (a b c : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 5) (h3 : c = 2) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = Real.sqrt 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1286_128697


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l1286_128666

/-- The area of a stripe wrapping around a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h1 : diameter = 20) 
  (h2 : stripe_width = 4) 
  (h3 : revolutions = 3) : 
  stripe_width * revolutions * (π * diameter) = 240 * π := by
sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l1286_128666


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1286_128614

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | -2 < x ∧ x < 2}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1286_128614


namespace NUMINAMATH_CALUDE_octagon_diagonal_property_l1286_128621

theorem octagon_diagonal_property (x : ℕ) (h : x > 2) :
  (x * (x - 3)) / 2 = x + 2 * (x - 2) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonal_property_l1286_128621


namespace NUMINAMATH_CALUDE_sqrt_225_equals_15_l1286_128673

theorem sqrt_225_equals_15 : Real.sqrt 225 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_225_equals_15_l1286_128673


namespace NUMINAMATH_CALUDE_victor_trays_l1286_128600

/-- The number of trays Victor can carry per trip -/
def tray_capacity : ℕ := 7

/-- The number of trips Victor made -/
def num_trips : ℕ := 4

/-- The number of trays picked up from the second table -/
def trays_second_table : ℕ := 5

/-- The number of trays picked up from the first table -/
def trays_first_table : ℕ := tray_capacity * num_trips - trays_second_table

theorem victor_trays : trays_first_table = 23 := by
  sorry

end NUMINAMATH_CALUDE_victor_trays_l1286_128600


namespace NUMINAMATH_CALUDE_one_root_quadratic_sum_l1286_128661

theorem one_root_quadratic_sum (a b : ℝ) : 
  (∃! x : ℝ, x^2 + a*x + b = 0) → 
  (a = 2*b - 3) → 
  (∃ b₁ b₂ : ℝ, (b = b₁ ∨ b = b₂) ∧ b₁ + b₂ = 4) := by
sorry

end NUMINAMATH_CALUDE_one_root_quadratic_sum_l1286_128661


namespace NUMINAMATH_CALUDE_condition_equivalence_l1286_128620

theorem condition_equivalence (a b : ℝ) (h : |a| > |b|) :
  (a - b > 0) ↔ (a + b > 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_equivalence_l1286_128620


namespace NUMINAMATH_CALUDE_pages_read_on_fourth_day_l1286_128668

/-- Given a book with 354 pages, if a person reads 63 pages on day one,
    twice that amount on day two, and 10 more pages than day two on day three,
    then the number of pages read on day four is 29. -/
theorem pages_read_on_fourth_day
  (total_pages : ℕ)
  (pages_day_one : ℕ)
  (h1 : total_pages = 354)
  (h2 : pages_day_one = 63)
  : total_pages - pages_day_one - (2 * pages_day_one) - (2 * pages_day_one + 10) = 29 := by
  sorry

#check pages_read_on_fourth_day

end NUMINAMATH_CALUDE_pages_read_on_fourth_day_l1286_128668


namespace NUMINAMATH_CALUDE_expression_evaluation_l1286_128619

theorem expression_evaluation :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) /
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1286_128619


namespace NUMINAMATH_CALUDE_train_length_l1286_128630

/-- The length of a train given its speed, bridge crossing time, and bridge length -/
theorem train_length (speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 275 →
  speed * crossing_time - bridge_length = 475 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1286_128630


namespace NUMINAMATH_CALUDE_proposition_truth_values_l1286_128626

theorem proposition_truth_values :
  let p := ∀ x y : ℝ, x > y → -x < -y
  let q := ∀ x y : ℝ, x > y → x^2 > y^2
  (p ∨ q) ∧ (p ∧ (¬q)) ∧ ¬(p ∧ q) ∧ ¬((¬p) ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l1286_128626


namespace NUMINAMATH_CALUDE_solution_satisfies_system_solution_is_unique_l1286_128687

/-- A system of linear equations with three variables -/
structure LinearSystem where
  eq1 : ℝ → ℝ → ℝ → Prop
  eq2 : ℝ → ℝ → ℝ → Prop
  eq3 : ℝ → ℝ → ℝ → Prop

/-- The specific system of equations from the problem -/
def problemSystem : LinearSystem where
  eq1 := fun x y z => x + y + z = 15
  eq2 := fun x y z => x - y + z = 5
  eq3 := fun x y z => x + y - z = 10

/-- The solution to the system of equations -/
def solution : ℝ × ℝ × ℝ := (7.5, 5, 2.5)

/-- Theorem stating that the solution satisfies the system of equations -/
theorem solution_satisfies_system :
  let (x, y, z) := solution
  problemSystem.eq1 x y z ∧
  problemSystem.eq2 x y z ∧
  problemSystem.eq3 x y z :=
by sorry

/-- Theorem stating that the solution is unique -/
theorem solution_is_unique :
  ∀ x y z, 
    problemSystem.eq1 x y z →
    problemSystem.eq2 x y z →
    problemSystem.eq3 x y z →
    (x, y, z) = solution :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_solution_is_unique_l1286_128687


namespace NUMINAMATH_CALUDE_milton_more_accelerated_l1286_128688

/-- Represents the percentage of at-home workforce for a city at a given year --/
structure WorkforceData :=
  (year2000 : ℝ)
  (year2010 : ℝ)
  (year2020 : ℝ)
  (year2030 : ℝ)

/-- Determines if a city's workforce growth is accelerating --/
def isAccelerating (data : WorkforceData) : Prop :=
  let diff2010 := data.year2010 - data.year2000
  let diff2020 := data.year2020 - data.year2010
  let diff2030 := data.year2030 - data.year2020
  diff2030 > diff2020 ∧ diff2020 > diff2010

/-- Milton City's workforce data --/
def miltonCity : WorkforceData :=
  { year2000 := 3
  , year2010 := 9
  , year2020 := 18
  , year2030 := 35 }

/-- Rivertown's workforce data --/
def rivertown : WorkforceData :=
  { year2000 := 4
  , year2010 := 7
  , year2020 := 13
  , year2030 := 20 }

/-- Theorem stating that Milton City's growth is more accelerated than Rivertown's --/
theorem milton_more_accelerated :
  isAccelerating miltonCity ∧ ¬isAccelerating rivertown :=
sorry

end NUMINAMATH_CALUDE_milton_more_accelerated_l1286_128688


namespace NUMINAMATH_CALUDE_square_area_decrease_l1286_128646

theorem square_area_decrease (initial_side : ℝ) (decrease_percent : ℝ) : 
  initial_side = 9 ∧ decrease_percent = 20 →
  let new_side := initial_side * (1 - decrease_percent / 100)
  let initial_area := initial_side ^ 2
  let new_area := new_side ^ 2
  let area_decrease_percent := (initial_area - new_area) / initial_area * 100
  area_decrease_percent = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_decrease_l1286_128646


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l1286_128662

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l1286_128662


namespace NUMINAMATH_CALUDE_alcohol_dilution_l1286_128682

/-- Proves that adding 16 liters of water to 24 liters of a 90% alcohol solution
    results in a new mixture with 54% alcohol. -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 24 →
  initial_concentration = 0.90 →
  added_water = 16 →
  final_concentration = 0.54 →
  initial_volume * initial_concentration = 
    (initial_volume + added_water) * final_concentration :=
by
  sorry

#check alcohol_dilution

end NUMINAMATH_CALUDE_alcohol_dilution_l1286_128682


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l1286_128683

def old_salary : ℝ := 10000
def new_salary : ℝ := 10200

theorem salary_increase_percentage :
  (new_salary - old_salary) / old_salary * 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l1286_128683


namespace NUMINAMATH_CALUDE_concert_revenue_is_930_l1286_128622

/-- Calculates the total revenue for a concert given the number of tickets sold and their prices. -/
def concert_revenue (student_tickets : ℕ) (non_student_tickets : ℕ) (student_price : ℕ) (non_student_price : ℕ) : ℕ :=
  student_tickets * student_price + non_student_tickets * non_student_price

/-- Proves that the total revenue for the concert is $930 given the specified conditions. -/
theorem concert_revenue_is_930 :
  let total_tickets : ℕ := 150
  let student_price : ℕ := 5
  let non_student_price : ℕ := 8
  let student_tickets : ℕ := 90
  let non_student_tickets : ℕ := 60
  concert_revenue student_tickets non_student_tickets student_price non_student_price = 930 :=
by
  sorry

#eval concert_revenue 90 60 5 8

end NUMINAMATH_CALUDE_concert_revenue_is_930_l1286_128622


namespace NUMINAMATH_CALUDE_triangle_property_l1286_128636

-- Define a structure for the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h : t.b + t.c = t.a * (Real.cos t.B + Real.cos t.C)) :
  t.A = Real.pi / 2 ∧ 
  Real.sqrt 3 + 2 < 2 * Real.cos (t.B / 2)^2 + 2 * Real.sqrt 3 * Real.cos (t.C / 2)^2 ∧
  2 * Real.cos (t.B / 2)^2 + 2 * Real.sqrt 3 * Real.cos (t.C / 2)^2 ≤ Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1286_128636


namespace NUMINAMATH_CALUDE_exam_score_difference_l1286_128615

def math_exam_problem (bryan_score jen_score sammy_score total_points sammy_mistakes : ℕ) : Prop :=
  bryan_score = 20 ∧
  jen_score = bryan_score + 10 ∧
  sammy_score < jen_score ∧
  total_points = 35 ∧
  sammy_mistakes = 7 ∧
  sammy_score = total_points - sammy_mistakes ∧
  jen_score - sammy_score = 2

theorem exam_score_difference :
  ∀ (bryan_score jen_score sammy_score total_points sammy_mistakes : ℕ),
  math_exam_problem bryan_score jen_score sammy_score total_points sammy_mistakes :=
by
  sorry

#check exam_score_difference

end NUMINAMATH_CALUDE_exam_score_difference_l1286_128615


namespace NUMINAMATH_CALUDE_cylinder_cube_surface_equality_l1286_128627

theorem cylinder_cube_surface_equality (r h s K : ℝ) : 
  r = 3 → h = 4 → 
  2 * π * r * h = 6 * s^2 → 
  s^3 = 48 / Real.sqrt K → 
  K = 36 / π^3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_cube_surface_equality_l1286_128627


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1286_128605

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1286_128605


namespace NUMINAMATH_CALUDE_trees_left_l1286_128699

theorem trees_left (initial_trees dead_trees : ℕ) 
  (h1 : initial_trees = 150) 
  (h2 : dead_trees = 24) : 
  initial_trees - dead_trees = 126 := by
sorry

end NUMINAMATH_CALUDE_trees_left_l1286_128699


namespace NUMINAMATH_CALUDE_waiter_tables_l1286_128670

/-- Given a waiter's customer and table information, prove the number of tables. -/
theorem waiter_tables
  (initial_customers : ℕ)
  (departed_customers : ℕ)
  (people_per_table : ℕ)
  (h1 : initial_customers = 44)
  (h2 : departed_customers = 12)
  (h3 : people_per_table = 8)
  : (initial_customers - departed_customers) / people_per_table = 4 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_l1286_128670


namespace NUMINAMATH_CALUDE_jane_initial_pick_is_one_fourth_l1286_128675

/-- The fraction of tomatoes Jane initially picked from a tomato plant -/
def jane_initial_pick : ℚ :=
  let initial_tomatoes : ℕ := 100
  let second_pick : ℕ := 20
  let third_pick : ℕ := 2 * second_pick
  let remaining_tomatoes : ℕ := 15
  (initial_tomatoes - second_pick - third_pick - remaining_tomatoes) / initial_tomatoes

theorem jane_initial_pick_is_one_fourth :
  jane_initial_pick = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_jane_initial_pick_is_one_fourth_l1286_128675


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l1286_128679

theorem quadratic_root_implies_coefficient (a : ℝ) : 
  (3 : ℝ)^2 + a * 3 + 9 = 0 → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l1286_128679


namespace NUMINAMATH_CALUDE_square_root_sum_equals_six_l1286_128650

theorem square_root_sum_equals_six : 
  Real.sqrt (15 - 6 * Real.sqrt 6) + Real.sqrt (15 + 6 * Real.sqrt 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_six_l1286_128650


namespace NUMINAMATH_CALUDE_adams_change_l1286_128689

/-- Given that Adam has $5 and an airplane costs $4.28, prove that he will receive $0.72 in change. -/
theorem adams_change (adams_money : ℚ) (airplane_cost : ℚ) (h1 : adams_money = 5) (h2 : airplane_cost = 4.28) :
  adams_money - airplane_cost = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_adams_change_l1286_128689


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l1286_128665

theorem stratified_sampling_medium_stores
  (total_stores : ℕ)
  (medium_stores : ℕ)
  (sample_size : ℕ)
  (h1 : total_stores = 300)
  (h2 : medium_stores = 75)
  (h3 : sample_size = 20) :
  (medium_stores : ℚ) / total_stores * sample_size = 5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l1286_128665


namespace NUMINAMATH_CALUDE_symmetric_points_relation_l1286_128613

/-- 
Given two points P and Q in the 2D plane, where:
- P has coordinates (m+1, 3)
- Q has coordinates (1, n-2)
- P is symmetric to Q with respect to the x-axis

This theorem proves that m-n = 1.
-/
theorem symmetric_points_relation (m n : ℝ) : 
  (∃ (P Q : ℝ × ℝ), 
    P = (m + 1, 3) ∧ 
    Q = (1, n - 2) ∧ 
    P.1 = Q.1 ∧ 
    P.2 = -Q.2) → 
  m - n = 1 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_relation_l1286_128613


namespace NUMINAMATH_CALUDE_meat_calculation_l1286_128671

/-- Given an initial amount of meat, calculate the remaining amount after using some for meatballs and spring rolls. -/
def remaining_meat (initial : ℝ) (meatball_fraction : ℝ) (spring_roll_amount : ℝ) : ℝ :=
  initial - (initial * meatball_fraction) - spring_roll_amount

/-- Theorem stating that given 20 kg of meat, using 1/4 for meatballs and 3 kg for spring rolls leaves 12 kg. -/
theorem meat_calculation :
  remaining_meat 20 (1/4) 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_meat_calculation_l1286_128671


namespace NUMINAMATH_CALUDE_total_peaches_l1286_128691

theorem total_peaches (red_peaches green_peaches : ℕ) 
  (h1 : red_peaches = 13) 
  (h2 : green_peaches = 3) : 
  red_peaches + green_peaches = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l1286_128691


namespace NUMINAMATH_CALUDE_log_equation_solutions_l1286_128654

theorem log_equation_solutions :
  ∀ x y : ℝ, x > 0 → y > 0 →
  (Real.log x / Real.log 4 - Real.log y / Real.log 2 = 0) →
  (x^2 - 5*y^2 + 4 = 0) →
  ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 2)) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solutions_l1286_128654


namespace NUMINAMATH_CALUDE_dans_minimum_speed_l1286_128649

/-- Proves that Dan must travel at a speed greater than 48 miles per hour to arrive in city B before Cara. -/
theorem dans_minimum_speed (distance : ℝ) (cara_speed : ℝ) (dan_delay : ℝ) : 
  distance = 120 → 
  cara_speed = 30 → 
  dan_delay = 1.5 → 
  ∀ dan_speed : ℝ, dan_speed > 48 → distance / dan_speed < distance / cara_speed - dan_delay := by
  sorry

#check dans_minimum_speed

end NUMINAMATH_CALUDE_dans_minimum_speed_l1286_128649


namespace NUMINAMATH_CALUDE_total_bugs_equals_63_l1286_128608

/-- The number of bugs eaten by the gecko -/
def gecko_bugs : ℕ := 12

/-- The number of bugs eaten by the lizard -/
def lizard_bugs : ℕ := gecko_bugs / 2

/-- The number of bugs eaten by the frog -/
def frog_bugs : ℕ := lizard_bugs * 3

/-- The number of bugs eaten by the toad -/
def toad_bugs : ℕ := frog_bugs + frog_bugs / 2

/-- The total number of bugs eaten by all animals -/
def total_bugs : ℕ := gecko_bugs + lizard_bugs + frog_bugs + toad_bugs

theorem total_bugs_equals_63 : total_bugs = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_bugs_equals_63_l1286_128608


namespace NUMINAMATH_CALUDE_divisible_by_nine_exists_l1286_128617

def is_distinct (digits : List Nat) : Prop :=
  digits.length = digits.toFinset.card

def sum_digits (digits : List Nat) : Nat :=
  digits.sum

theorem divisible_by_nine_exists (kolya_number : List Nat) :
  kolya_number.length = 10 →
  (∀ d ∈ kolya_number, d < 10) →
  is_distinct kolya_number →
  ∃ d : Nat, d < 10 ∧ d ∉ kolya_number ∧
    (sum_digits kolya_number + d) % 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_nine_exists_l1286_128617


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1286_128663

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum_345 : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1286_128663


namespace NUMINAMATH_CALUDE_solve_equation_l1286_128618

theorem solve_equation (x : ℝ) : 3*x + 15 = (1/3) * (8*x + 48) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1286_128618


namespace NUMINAMATH_CALUDE_sum_of_digits_l1286_128602

theorem sum_of_digits (a b c d : ℕ) : 
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  100 * a + 10 * b + c + 100 * d + 10 * c + a = 1100 →
  a + b + c + d = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l1286_128602


namespace NUMINAMATH_CALUDE_specific_ap_first_term_l1286_128645

/-- An arithmetic progression with given parameters -/
structure ArithmeticProgression where
  n : ℕ             -- number of terms
  d : ℤ             -- common difference
  last_term : ℤ     -- last term

/-- The first term of an arithmetic progression -/
def first_term (ap : ArithmeticProgression) : ℤ :=
  ap.last_term - (ap.n - 1) * ap.d

/-- Theorem stating the first term of the specific arithmetic progression -/
theorem specific_ap_first_term :
  let ap : ArithmeticProgression := ⟨31, 2, 62⟩
  first_term ap = 2 := by sorry

end NUMINAMATH_CALUDE_specific_ap_first_term_l1286_128645


namespace NUMINAMATH_CALUDE_triangle_properties_l1286_128638

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < π / 2
  acute_B : 0 < B ∧ B < π / 2
  acute_C : 0 < C ∧ C < π / 2
  sum_angles : A + B + C = π

-- Define the theorem
theorem triangle_properties (t : AcuteTriangle) 
  (h1 : Real.sin (t.A + t.B) = 3/5)
  (h2 : Real.sin (t.A - t.B) = 1/5)
  (h3 : ∃ (AB : Real), AB = 3) :
  (Real.tan t.A = 2 * Real.tan t.B) ∧
  (∃ (height : Real), height = 2 + Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1286_128638


namespace NUMINAMATH_CALUDE_thirteen_ceilings_left_l1286_128658

/-- Represents the number of ceilings left to paint after next week -/
def ceilings_left_after_next_week (stories : ℕ) (rooms_per_floor : ℕ) (ceilings_painted_this_week : ℕ) : ℕ :=
  let total_room_ceilings := stories * rooms_per_floor
  let total_hallway_ceilings := stories
  let total_ceilings := total_room_ceilings + total_hallway_ceilings
  let ceilings_left_after_this_week := total_ceilings - ceilings_painted_this_week
  let ceilings_to_paint_next_week := ceilings_painted_this_week / 4 + stories
  ceilings_left_after_this_week - ceilings_to_paint_next_week

/-- Theorem stating that 13 ceilings will be left to paint after next week -/
theorem thirteen_ceilings_left : ceilings_left_after_next_week 4 7 12 = 13 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_ceilings_left_l1286_128658


namespace NUMINAMATH_CALUDE_quadratic_equation_with_prime_roots_l1286_128641

theorem quadratic_equation_with_prime_roots (a m : ℤ) :
  (∃ x y : ℕ, x ≠ y ∧ Prime x ∧ Prime y ∧ (a * x^2 - m * x + 1996 = 0) ∧ (a * y^2 - m * y + 1996 = 0)) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_prime_roots_l1286_128641


namespace NUMINAMATH_CALUDE_same_solution_implies_m_value_l1286_128656

theorem same_solution_implies_m_value : ∀ m x : ℚ,
  (8 - m = 2 * (x + 1) ∧ 2 * (2 * x - 3) - 1 = 1 - 2 * x) →
  m = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_m_value_l1286_128656


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_for_problem_config_l1286_128667

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- The maximum squared radius of a sphere that can fit within two intersecting cones -/
def maxSphereRadiusSquared (ic : IntersectingCones) : ℝ := sorry

/-- The specific configuration described in the problem -/
def problemConfig : IntersectingCones :=
  { cone1 := { baseRadius := 4, height := 10 }
  , cone2 := { baseRadius := 4, height := 10 }
  , intersectionDistance := 4
  }

theorem max_sphere_radius_squared_for_problem_config :
  maxSphereRadiusSquared problemConfig = 4176 / 841 :=
sorry

end NUMINAMATH_CALUDE_max_sphere_radius_squared_for_problem_config_l1286_128667


namespace NUMINAMATH_CALUDE_modulo_17_residue_l1286_128625

theorem modulo_17_residue : (3^4 + 6 * 49 + 8 * 137 + 7 * 34) % 17 = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulo_17_residue_l1286_128625


namespace NUMINAMATH_CALUDE_doll_count_l1286_128629

/-- The number of dolls Ivy has -/
def ivy_dolls : ℕ := 30

/-- The number of dolls Dina has -/
def dina_dolls : ℕ := 2 * ivy_dolls

/-- The number of collector's edition dolls Ivy has -/
def ivy_collector_dolls : ℕ := 20

/-- The number of dolls Casey has -/
def casey_dolls : ℕ := 5 * ivy_collector_dolls

/-- The total number of dolls Dina, Ivy, and Casey have together -/
def total_dolls : ℕ := dina_dolls + ivy_dolls + casey_dolls

theorem doll_count : total_dolls = 190 ∧ 
  2 * ivy_dolls / 3 = ivy_collector_dolls := by
  sorry

end NUMINAMATH_CALUDE_doll_count_l1286_128629


namespace NUMINAMATH_CALUDE_four_Z_three_l1286_128652

def Z (a b : ℤ) : ℤ := a^2 - 3*a*b + b^2

theorem four_Z_three : Z 4 3 = -11 := by
  sorry

end NUMINAMATH_CALUDE_four_Z_three_l1286_128652


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1286_128651

theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := first_half_time * second_half_time_factor
  let total_time := first_half_time + second_half_time
  (total_distance / total_time) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1286_128651


namespace NUMINAMATH_CALUDE_diamond_operation_l1286_128677

def diamond (a b : ℤ) : ℤ := 12 * a - 10 * b

theorem diamond_operation : diamond (diamond (diamond (diamond 20 22) 22) 22) 22 = 20 := by
  sorry

end NUMINAMATH_CALUDE_diamond_operation_l1286_128677


namespace NUMINAMATH_CALUDE_shirt_fabric_sum_l1286_128678

theorem shirt_fabric_sum (a : ℝ) (r : ℝ) (h1 : a = 2011) (h2 : r = 4/5) (h3 : r < 1) :
  a / (1 - r) = 10055 := by
  sorry

end NUMINAMATH_CALUDE_shirt_fabric_sum_l1286_128678


namespace NUMINAMATH_CALUDE_multitool_comparison_l1286_128644

-- Define the contents of each multitool
def walmart_tools : ℕ := 2 + 4 + 1 + 1 + 1
def target_knives : ℕ := 4 * 3
def target_tools : ℕ := 3 + target_knives + 2 + 1 + 1 + 2

-- Theorem to prove the difference in tools and the ratio
theorem multitool_comparison :
  (target_tools - walmart_tools = 12) ∧
  (target_tools / walmart_tools = 7 / 3) := by
  sorry

#eval walmart_tools
#eval target_tools

end NUMINAMATH_CALUDE_multitool_comparison_l1286_128644


namespace NUMINAMATH_CALUDE_white_balls_count_l1286_128612

theorem white_balls_count (black_balls : ℕ) (prob_white : ℚ) (white_balls : ℕ) : 
  black_balls = 6 →
  prob_white = 45454545454545453 / 100000000000000000 →
  (white_balls : ℚ) / ((black_balls : ℚ) + (white_balls : ℚ)) = prob_white →
  white_balls = 5 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l1286_128612


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1286_128648

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x * (x - 3) - (x - 3)
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1286_128648


namespace NUMINAMATH_CALUDE_square_root_of_nine_three_is_square_root_of_nine_l1286_128607

theorem square_root_of_nine (x : ℝ) : x ^ 2 = 9 → x = 3 ∨ x = -3 := by
  sorry

theorem three_is_square_root_of_nine : ∃ x : ℝ, x ^ 2 = 9 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_three_is_square_root_of_nine_l1286_128607


namespace NUMINAMATH_CALUDE_domain_intersection_l1286_128659

-- Define the domain of y = e^x
def M : Set ℝ := {y | ∃ x, y = Real.exp x}

-- Define the domain of y = ln x
def N : Set ℝ := {y | ∃ x, y = Real.log x}

-- Theorem statement
theorem domain_intersection :
  M ∩ N = {y : ℝ | y > 0} := by sorry

end NUMINAMATH_CALUDE_domain_intersection_l1286_128659


namespace NUMINAMATH_CALUDE_probability_of_selecting_A_or_B_l1286_128611

-- Define the total number of experts
def total_experts : ℕ := 6

-- Define the number of experts to be selected
def selected_experts : ℕ := 2

-- Define the probability of selecting at least one of A or B
def prob_select_A_or_B : ℚ := 3/5

-- Theorem statement
theorem probability_of_selecting_A_or_B :
  let total_combinations := Nat.choose total_experts selected_experts
  let combinations_without_A_and_B := Nat.choose (total_experts - 2) selected_experts
  1 - (combinations_without_A_and_B : ℚ) / total_combinations = prob_select_A_or_B :=
by sorry

end NUMINAMATH_CALUDE_probability_of_selecting_A_or_B_l1286_128611


namespace NUMINAMATH_CALUDE_no_real_solutions_l1286_128694

theorem no_real_solutions :
  ¬∃ (x y z u : ℝ), x^4 - 17 = y^4 - 7 ∧ 
                    x^4 - 17 = z^4 + 19 ∧ 
                    x^4 - 17 = u^4 + 5 ∧ 
                    x^4 - 17 = x * y * z * u :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1286_128694


namespace NUMINAMATH_CALUDE_max_bribe_amount_l1286_128603

/-- Represents the bet amount in coins -/
def betAmount : ℤ := 100

/-- Represents the maximum bribe amount in coins -/
def maxBribe : ℤ := 199

/-- 
Proves that the maximum bribe a person would pay to avoid eviction is 199 coins, 
given a bet where they lose 100 coins if evicted and gain 100 coins if not evicted, 
assuming they act solely in their own financial interest.
-/
theorem max_bribe_amount : 
  ∀ (bribe : ℤ), 
    bribe ≤ maxBribe ∧ 
    bribe > betAmount ∧
    (maxBribe - betAmount ≤ betAmount) ∧
    (∀ (x : ℤ), x > maxBribe → x - betAmount > betAmount) := by
  sorry


end NUMINAMATH_CALUDE_max_bribe_amount_l1286_128603


namespace NUMINAMATH_CALUDE_sam_memorized_digits_l1286_128684

/-- Given information about the number of digits of pi memorized by Sam, Carlos, and Mina,
    prove that Sam memorized 10 digits. -/
theorem sam_memorized_digits (sam carlos mina : ℕ) 
  (h1 : sam = carlos + 6)
  (h2 : mina = 6 * carlos)
  (h3 : mina = 24) : 
  sam = 10 := by sorry

end NUMINAMATH_CALUDE_sam_memorized_digits_l1286_128684


namespace NUMINAMATH_CALUDE_no_solution_equation_l1286_128633

theorem no_solution_equation : ¬∃ (x : ℝ), x ≠ 2 ∧ x + 5 / (x - 2) = 2 + 5 / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l1286_128633


namespace NUMINAMATH_CALUDE_factorial_divisibility_l1286_128693

theorem factorial_divisibility (m n : ℕ) : 
  (m.factorial * n.factorial * (m + n).factorial) ∣ ((2 * m).factorial * (2 * n).factorial) := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l1286_128693


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l1286_128685

theorem unfair_coin_probability (p : ℝ) : 
  0 < p ∧ p < 1 →
  (6 : ℝ) * p^2 * (1 - p)^2 = (4 : ℝ) * p^3 * (1 - p) →
  p = 3/5 := by
sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l1286_128685


namespace NUMINAMATH_CALUDE_exists_special_function_l1286_128696

theorem exists_special_function : ∃ (f : ℝ → ℝ),
  (∀ (b : ℝ), ∃! (x : ℝ), f x = b) ∧
  (∀ (a b : ℝ), a > 0 → ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a * x₁ + b ∧ f x₂ = a * x₂ + b) :=
by sorry

end NUMINAMATH_CALUDE_exists_special_function_l1286_128696


namespace NUMINAMATH_CALUDE_lowest_number_in_range_l1286_128639

/-- The probability of selecting a number greater than another randomly selected number -/
def probability : ℚ := 4995 / 10000

/-- Theorem stating that given the probability, the lowest number in the range is 999 -/
theorem lowest_number_in_range (x y : ℕ) (h : x ≤ y) :
  (((y - x) * (y - x + 1) : ℚ) / (2 * (y - x + 1)^2)) = probability → x = 999 := by
  sorry

end NUMINAMATH_CALUDE_lowest_number_in_range_l1286_128639


namespace NUMINAMATH_CALUDE_janes_mean_score_l1286_128635

def quiz_scores : List ℝ := [99, 95, 93, 87, 90]
def exam_scores : List ℝ := [88, 92]

def all_scores : List ℝ := quiz_scores ++ exam_scores

theorem janes_mean_score :
  (all_scores.sum / all_scores.length : ℝ) = 644 / 7 := by
  sorry

end NUMINAMATH_CALUDE_janes_mean_score_l1286_128635


namespace NUMINAMATH_CALUDE_smallest_marble_count_thirty_is_smallest_l1286_128676

theorem smallest_marble_count : ℕ → Prop :=
  fun n => n > 0 ∧ 
    (∃ w g r b : ℕ, 
      w + g + r + b = n ∧ 
      w = n / 6 ∧ 
      g = n / 5 ∧ 
      r + b = 19 * n / 30) →
  n ≥ 30

theorem thirty_is_smallest : smallest_marble_count 30 :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_thirty_is_smallest_l1286_128676


namespace NUMINAMATH_CALUDE_school_distance_l1286_128610

/-- 
Given a person who walks to and from a destination for 5 days, 
with an additional 4km on the last day, and a total distance of 74km,
prove that the one-way distance to the destination is 7km.
-/
theorem school_distance (x : ℝ) 
  (h1 : (4 * 2 * x) + (2 * x + 4) = 74) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_school_distance_l1286_128610


namespace NUMINAMATH_CALUDE_sin_equality_proof_l1286_128655

theorem sin_equality_proof (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 → Real.sin (n * π / 180) = Real.sin (721 * π / 180) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_proof_l1286_128655


namespace NUMINAMATH_CALUDE_merchant_discount_l1286_128695

theorem merchant_discount (markup : ℝ) (profit : ℝ) (discount : ℝ) : 
  markup = 0.75 → 
  profit = 0.225 → 
  discount = (markup + 1 - (profit + 1)) / (markup + 1) * 100 →
  discount = 30 := by
  sorry

end NUMINAMATH_CALUDE_merchant_discount_l1286_128695
