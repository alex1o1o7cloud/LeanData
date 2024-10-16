import Mathlib

namespace NUMINAMATH_CALUDE_product_greater_than_sum_minus_one_l2117_211757

theorem product_greater_than_sum_minus_one {a₁ a₂ : ℝ} 
  (h₁ : 0 < a₁) (h₂ : a₁ < 1) (h₃ : 0 < a₂) (h₄ : a₂ < 1) : 
  a₁ * a₂ > a₁ + a₂ - 1 := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_minus_one_l2117_211757


namespace NUMINAMATH_CALUDE_asterisk_replacement_l2117_211760

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 21) * (x / 189) = 1 := by sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l2117_211760


namespace NUMINAMATH_CALUDE_tan_sum_alpha_beta_l2117_211715

theorem tan_sum_alpha_beta (α β : Real) (h : 2 * Real.tan α = 3 * Real.tan β) :
  Real.tan (α + β) = (5 * Real.sin (2 * β)) / (5 * Real.cos (2 * β) - 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_alpha_beta_l2117_211715


namespace NUMINAMATH_CALUDE_students_not_playing_sports_l2117_211751

theorem students_not_playing_sports (total : ℕ) (basketball : ℕ) (volleyball : ℕ) (both : ℕ) : 
  total = 20 ∧ 
  basketball = total / 2 ∧ 
  volleyball = total * 2 / 5 ∧ 
  both = total / 10 → 
  total - (basketball + volleyball - both) = 4 :=
by sorry

end NUMINAMATH_CALUDE_students_not_playing_sports_l2117_211751


namespace NUMINAMATH_CALUDE_sqrt_65_bounds_l2117_211704

theorem sqrt_65_bounds (n : ℕ+) : n < Real.sqrt 65 ∧ Real.sqrt 65 < n + 1 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_65_bounds_l2117_211704


namespace NUMINAMATH_CALUDE_inequality_proof_l2117_211792

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2117_211792


namespace NUMINAMATH_CALUDE_sum_of_coordinates_is_16_l2117_211752

/-- Given two points A and B in a 2D plane, where:
  - A is at the origin (0, 0)
  - B is on the line y = 6
  - The slope of segment AB is 3/5
  Prove that the sum of the x- and y-coordinates of B is 16. -/
theorem sum_of_coordinates_is_16 (B : ℝ × ℝ) : 
  B.2 = 6 ∧ 
  (B.2 - 0) / (B.1 - 0) = 3 / 5 → 
  B.1 + B.2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_is_16_l2117_211752


namespace NUMINAMATH_CALUDE_cubic_equation_transformation_l2117_211769

theorem cubic_equation_transformation (A B C : ℝ) :
  ∃ (p q β : ℝ), ∀ (z x : ℝ),
    (z^3 + A * z^2 + B * z + C = 0) ↔
    (z = x + β ∧ x^3 + p * x + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_transformation_l2117_211769


namespace NUMINAMATH_CALUDE_constant_phi_is_cone_l2117_211722

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the set of points satisfying φ = c
def ConstantPhiSet (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

-- Define a cone (we'll use a simplified definition for this statement)
def Cone : Set SphericalCoord := sorry

-- Theorem statement
theorem constant_phi_is_cone (c : ℝ) : 
  ConstantPhiSet c = Cone := by sorry

end NUMINAMATH_CALUDE_constant_phi_is_cone_l2117_211722


namespace NUMINAMATH_CALUDE_custom_mult_solution_l2117_211747

/-- Custom multiplication operation -/
def custom_mult (a b : ℚ) : ℚ := 3 * a - 2 * b^2

/-- Theorem stating that if a * 4 = -7 using the custom multiplication, then a = 25/3 -/
theorem custom_mult_solution :
  ∀ a : ℚ, custom_mult a 4 = -7 → a = 25/3 := by
sorry

end NUMINAMATH_CALUDE_custom_mult_solution_l2117_211747


namespace NUMINAMATH_CALUDE_age_difference_l2117_211731

theorem age_difference (a b c : ℕ) : 
  b = 20 →
  c = b / 2 →
  a + b + c = 52 →
  a = b + 2 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2117_211731


namespace NUMINAMATH_CALUDE_min_cut_length_40x30_paper_l2117_211770

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the problem setup -/
structure PaperCutProblem where
  paper : Rectangle
  inner_rectangle : Rectangle
  num_cuts : ℕ

/-- The minimum total length of cuts for the given problem -/
def min_cut_length (problem : PaperCutProblem) : ℕ := 
  2 * problem.paper.width + 2 * problem.paper.height

/-- Theorem stating the minimum cut length for the specific problem -/
theorem min_cut_length_40x30_paper (problem : PaperCutProblem) 
  (h1 : problem.paper = ⟨40, 30⟩) 
  (h2 : problem.inner_rectangle = ⟨10, 5⟩) 
  (h3 : problem.num_cuts = 4) : 
  min_cut_length problem = 140 := by
  sorry

#check min_cut_length_40x30_paper

end NUMINAMATH_CALUDE_min_cut_length_40x30_paper_l2117_211770


namespace NUMINAMATH_CALUDE_three_prime_divisors_l2117_211717

theorem three_prime_divisors (p : Nat) (h_prime : Prime p) 
  (h_cong : (2^(p-1)) % (p^2) = 1) (n : Nat) : 
  ∃ (q₁ q₂ q₃ : Nat), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ 
  q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₂ ≠ q₃ ∧
  (q₁ ∣ ((p-1) * (Nat.factorial p + 2^n))) ∧
  (q₂ ∣ ((p-1) * (Nat.factorial p + 2^n))) ∧
  (q₃ ∣ ((p-1) * (Nat.factorial p + 2^n))) := by
sorry

end NUMINAMATH_CALUDE_three_prime_divisors_l2117_211717


namespace NUMINAMATH_CALUDE_time_to_write_rearrangements_l2117_211748

def name_length : ℕ := 5
def rearrangements_per_minute : ℕ := 20

theorem time_to_write_rearrangements :
  (Nat.factorial name_length / rearrangements_per_minute : ℚ) / 60 = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_time_to_write_rearrangements_l2117_211748


namespace NUMINAMATH_CALUDE_equation_value_proof_l2117_211741

theorem equation_value_proof (x y z w : ℝ) 
  (eq1 : 4 * x * z + y * w = 4)
  (eq2 : (2 * x + y) * (2 * z + w) = 20) :
  x * w + y * z = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_proof_l2117_211741


namespace NUMINAMATH_CALUDE_fourth_power_sum_l2117_211740

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a^2 + b^2 + c^2 = 5) 
  (h3 : a^3 + b^3 + c^3 = 8) : 
  a^4 + b^4 + c^4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l2117_211740


namespace NUMINAMATH_CALUDE_trig_identity_proof_l2117_211719

theorem trig_identity_proof (α : Real) (h : Real.tan α = 2) :
  4 * (Real.sin α)^2 - 3 * (Real.sin α) * (Real.cos α) - 5 * (Real.cos α)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2117_211719


namespace NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l2117_211762

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if the perimeter is equal to 3(sin A + sin B + sin C),
    then the diameter of its circumcircle is 3. -/
theorem triangle_circumcircle_diameter
  (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a + b + c = 3 * (Real.sin A + Real.sin B + Real.sin C) →
  a / Real.sin A = 2 * R →
  b / Real.sin B = 2 * R →
  c / Real.sin C = 2 * R →
  2 * R = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l2117_211762


namespace NUMINAMATH_CALUDE_yellow_jelly_bean_probability_l2117_211799

theorem yellow_jelly_bean_probability 
  (red_prob : ℝ) 
  (orange_prob : ℝ) 
  (blue_prob : ℝ) 
  (yellow_prob : ℝ)
  (h1 : red_prob = 0.1)
  (h2 : orange_prob = 0.4)
  (h3 : blue_prob = 0.2)
  (h4 : red_prob + orange_prob + blue_prob + yellow_prob = 1) :
  yellow_prob = 0.3 := by
sorry

end NUMINAMATH_CALUDE_yellow_jelly_bean_probability_l2117_211799


namespace NUMINAMATH_CALUDE_race_equation_theorem_l2117_211745

/-- Represents a runner's performance in a race before and after training -/
structure RunnerPerformance where
  distance : ℝ
  speedIncrease : ℝ
  timeImprovement : ℝ
  initialSpeed : ℝ

/-- Checks if the given runner performance satisfies the race equation -/
def satisfiesRaceEquation (perf : RunnerPerformance) : Prop :=
  perf.distance / perf.initialSpeed - 
  perf.distance / (perf.initialSpeed * (1 + perf.speedIncrease)) = 
  perf.timeImprovement

/-- Theorem stating that a runner with the given performance satisfies the race equation -/
theorem race_equation_theorem (perf : RunnerPerformance) 
  (h1 : perf.distance = 3000)
  (h2 : perf.speedIncrease = 0.25)
  (h3 : perf.timeImprovement = 3) :
  satisfiesRaceEquation perf := by
  sorry

end NUMINAMATH_CALUDE_race_equation_theorem_l2117_211745


namespace NUMINAMATH_CALUDE_angle_WYZ_measure_l2117_211714

-- Define the angles in degrees
def angle_XYZ : ℝ := 45
def angle_XYW : ℝ := 15

-- Define the theorem
theorem angle_WYZ_measure : 
  let angle_WYZ := angle_XYZ - angle_XYW
  angle_WYZ = 30 := by sorry

end NUMINAMATH_CALUDE_angle_WYZ_measure_l2117_211714


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l2117_211788

/-- The amount of money Chris received from his grandmother -/
def grandmother_gift : ℕ := 25

/-- The amount of money Chris received from his aunt and uncle -/
def aunt_uncle_gift : ℕ := 20

/-- The amount of money Chris received from his parents -/
def parents_gift : ℕ := 75

/-- The total amount of money Chris has after his birthday -/
def total_after_birthday : ℕ := 279

/-- The amount of money Chris had before his birthday -/
def amount_before_birthday : ℕ := total_after_birthday - (grandmother_gift + aunt_uncle_gift + parents_gift)

theorem chris_money_before_birthday :
  amount_before_birthday = 159 :=
by sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l2117_211788


namespace NUMINAMATH_CALUDE_mike_ride_distance_l2117_211796

/-- Represents the taxi fare structure and ride details -/
structure TaxiRide where
  start_fee : ℝ
  per_mile_fee : ℝ
  toll_fee : ℝ
  distance : ℝ

/-- Calculates the total fare for a taxi ride -/
def total_fare (ride : TaxiRide) : ℝ :=
  ride.start_fee + ride.toll_fee + ride.per_mile_fee * ride.distance

/-- Proves that Mike's ride was 34 miles long given the conditions -/
theorem mike_ride_distance :
  let mike : TaxiRide := { start_fee := 2.5, per_mile_fee := 0.25, toll_fee := 0, distance := 34 }
  let annie : TaxiRide := { start_fee := 2.5, per_mile_fee := 0.25, toll_fee := 5, distance := 14 }
  total_fare mike = total_fare annie := by
  sorry

#check mike_ride_distance

end NUMINAMATH_CALUDE_mike_ride_distance_l2117_211796


namespace NUMINAMATH_CALUDE_buses_meet_time_l2117_211746

/-- Represents a time of day in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : ℕ := sorry

/-- Represents a bus journey -/
structure BusJourney where
  startTime : Time
  endTime : Time
  distance : ℝ

theorem buses_meet_time
  (totalDistance : ℝ)
  (lishanToCounty : ℝ)
  (busAToCounty : BusJourney)
  (busAToProvincial : BusJourney)
  (busBSpeed : ℝ)
  (busBStartTime : Time)
  (h1 : totalDistance = 189)
  (h2 : lishanToCounty = 54)
  (h3 : busAToCounty.startTime = ⟨8, 30, sorry⟩)
  (h4 : busAToCounty.endTime = ⟨9, 15, sorry⟩)
  (h5 : busAToProvincial.startTime = ⟨9, 30, sorry⟩)
  (h6 : busAToProvincial.endTime = ⟨11, 0, sorry⟩)
  (h7 : busBStartTime = ⟨8, 50, sorry⟩)
  (h8 : busBSpeed = 60)
  (h9 : busAToCounty.distance = lishanToCounty)
  (h10 : busAToProvincial.distance = totalDistance - lishanToCounty) :
  ∃ (meetTime : Time), meetTime = ⟨10, 8, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_buses_meet_time_l2117_211746


namespace NUMINAMATH_CALUDE_min_value_C_squared_minus_D_squared_l2117_211780

theorem min_value_C_squared_minus_D_squared :
  ∀ (x y z : ℝ), 
  x ≥ 0 → y ≥ 0 → z ≥ 0 →
  x ≤ 1 → y ≤ 2 → z ≤ 3 →
  let C := Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12)
  let D := Real.sqrt (x + 1) + Real.sqrt (y + 2) + Real.sqrt (z + 3)
  ∀ (C' D' : ℝ), C = C' → D = D' →
  C' ^ 2 - D' ^ 2 ≥ 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_C_squared_minus_D_squared_l2117_211780


namespace NUMINAMATH_CALUDE_transylvanian_vampire_statement_l2117_211736

-- Define the possible species
inductive Species
| Human
| Vampire

-- Define the possible mental states
inductive MentalState
| Sane
| Insane

-- Define a person
structure Person where
  species : Species
  mentalState : MentalState

-- Define the statement made by the person
def madeVampireStatement (p : Person) : Prop :=
  p.mentalState = MentalState.Insane

-- Theorem statement
theorem transylvanian_vampire_statement 
  (p : Person) 
  (h : madeVampireStatement p) : 
  (∃ (s : Species), p.species = s) ∧ 
  (p.mentalState = MentalState.Insane) :=
sorry

end NUMINAMATH_CALUDE_transylvanian_vampire_statement_l2117_211736


namespace NUMINAMATH_CALUDE_min_balls_for_same_color_l2117_211700

def box : Finset (Fin 6) := Finset.univ
def color : Fin 6 → ℕ
  | 0 => 28  -- red
  | 1 => 20  -- green
  | 2 => 19  -- yellow
  | 3 => 13  -- blue
  | 4 => 11  -- white
  | 5 => 9   -- black

theorem min_balls_for_same_color : 
  ∀ n : ℕ, (∀ s : Finset (Fin 6), s.card = n → 
    (∃ c : Fin 6, (s.filter (λ i => color i = color c)).card < 15)) → 
  n < 76 :=
sorry

end NUMINAMATH_CALUDE_min_balls_for_same_color_l2117_211700


namespace NUMINAMATH_CALUDE_first_patient_therapy_hours_l2117_211758

/-- Represents the cost structure and patient charges for a psychologist's therapy sessions. -/
structure TherapyCost where
  first_hour : ℕ           -- Cost of the first hour
  additional_hour : ℕ      -- Cost of each additional hour
  first_patient_total : ℕ  -- Total charge for the first patient
  two_hour_total : ℕ       -- Total charge for a patient receiving 2 hours

/-- Calculates the number of therapy hours for the first patient given the cost structure. -/
def calculate_therapy_hours (cost : TherapyCost) : ℕ :=
  -- The implementation is not provided as per the instructions
  sorry

/-- Theorem stating that given the specific cost structure, the first patient received 5 hours of therapy. -/
theorem first_patient_therapy_hours 
  (cost : TherapyCost)
  (h1 : cost.first_hour = cost.additional_hour + 35)
  (h2 : cost.two_hour_total = 161)
  (h3 : cost.first_patient_total = 350) :
  calculate_therapy_hours cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_patient_therapy_hours_l2117_211758


namespace NUMINAMATH_CALUDE_f_inf_fixed_point_l2117_211721

variable {A : Type*} [Fintype A]
variable (f : A → A)

def f_n : ℕ → (Set A) → Set A
  | 0, S => S
  | n + 1, S => f '' (f_n n S)

def f_inf (S : Set A) : Set A :=
  ⋂ n, f_n f n S

theorem f_inf_fixed_point (S : Set A) :
  f '' (f_inf f S) = f_inf f S := by sorry

end NUMINAMATH_CALUDE_f_inf_fixed_point_l2117_211721


namespace NUMINAMATH_CALUDE_tangent_line_at_one_two_l2117_211782

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (-x - 1) - x
  else Real.exp (x - 1) + x

-- State the theorem
theorem tangent_line_at_one_two :
  (∀ x : ℝ, f x = f (-x)) → -- f is even
  f 1 = 2 → -- (1, 2) lies on the curve
  ∃ m : ℝ, ∀ x : ℝ, (HasDerivAt f m 1 ∧ m = 2) → 
    2 = m * (1 - 1) + f 1 ∧ -- Point-slope form at (1, 2)
    ∀ y : ℝ, y = 2 * x ↔ y - f 1 = m * (x - 1) -- Tangent line equation
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_two_l2117_211782


namespace NUMINAMATH_CALUDE_composite_quotient_l2117_211754

theorem composite_quotient (n : ℕ) (h1 : n ≥ 4) (h2 : n ∣ 2^n - 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (2^n - 2) / n = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_quotient_l2117_211754


namespace NUMINAMATH_CALUDE_initial_candies_equals_sum_of_given_and_left_l2117_211795

/-- Given the number of candies given away and the number of candies left,
    prove that the initial number of candies is their sum. -/
theorem initial_candies_equals_sum_of_given_and_left (given away : ℕ) (left : ℕ) :
  given + left = given + left := by sorry

end NUMINAMATH_CALUDE_initial_candies_equals_sum_of_given_and_left_l2117_211795


namespace NUMINAMATH_CALUDE_football_team_progress_l2117_211701

/-- Calculates the total progress of a football team given yards lost and gained -/
def footballProgress (yardsLost : Int) (yardsGained : Int) : Int :=
  yardsGained - yardsLost

/-- Theorem: A football team that lost 5 yards and then gained 11 yards has a total progress of 6 yards -/
theorem football_team_progress :
  footballProgress 5 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l2117_211701


namespace NUMINAMATH_CALUDE_min_distance_parallel_lines_l2117_211784

/-- The minimum distance between two parallel lines -/
theorem min_distance_parallel_lines :
  let line1 : ℝ → ℝ → Prop := λ x y => 3 * x + 4 * y - 6 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => 6 * x + 8 * y + 3 = 0
  ∀ (P Q : ℝ × ℝ), line1 P.1 P.2 → line2 Q.1 Q.2 →
  (∀ (P' Q' : ℝ × ℝ), line1 P'.1 P'.2 → line2 Q'.1 Q'.2 →
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2)) →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 3/2 :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_parallel_lines_l2117_211784


namespace NUMINAMATH_CALUDE_expand_product_l2117_211772

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x - 5) = 6 * x^2 - 7 * x - 20 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2117_211772


namespace NUMINAMATH_CALUDE_sum_of_first_10_common_elements_l2117_211763

/-- Arithmetic progression with first term 5 and common difference 3 -/
def ap (n : ℕ) : ℕ := 5 + 3 * n

/-- Geometric progression with first term 10 and common ratio 2 -/
def gp (k : ℕ) : ℕ := 10 * 2^k

/-- The sequence of common elements between ap and gp -/
def common_sequence (n : ℕ) : ℕ := 20 * 4^n

theorem sum_of_first_10_common_elements : 
  (Finset.range 10).sum common_sequence = 6990500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_10_common_elements_l2117_211763


namespace NUMINAMATH_CALUDE_certain_number_solution_l2117_211766

theorem certain_number_solution (x : ℝ) : 
  8 * 5.4 - (x * 10) / 1.2 = 31.000000000000004 → x = 1.464 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_solution_l2117_211766


namespace NUMINAMATH_CALUDE_product_digits_sum_base7_l2117_211723

/-- Converts a base 7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a number in base 7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits in base 7 of the product of 16₇ and 21₇ is equal to 3₇ --/
theorem product_digits_sum_base7 : 
  sumOfDigitsBase7 (toBase7 (toDecimal 16 * toDecimal 21)) = 3 := by sorry

end NUMINAMATH_CALUDE_product_digits_sum_base7_l2117_211723


namespace NUMINAMATH_CALUDE_winning_strategy_l2117_211777

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Predicate to check if a number is a Fibonacci number -/
def isFibonacci (n : ℕ) : Prop :=
  ∃ k, fib k = n

/-- Game rules -/
structure GameRules where
  n : ℕ
  n_gt_one : n > 1
  first_turn_not_all : ∀ (first_pick : ℕ), first_pick < n
  subsequent_turns : ∀ (prev_pick current_pick : ℕ), current_pick ≤ 2 * prev_pick

/-- Winning strategy for Player A -/
def playerAWins (rules : GameRules) : Prop :=
  ¬(isFibonacci rules.n)

/-- Main theorem: Player A has a winning strategy iff n is not a Fibonacci number -/
theorem winning_strategy (rules : GameRules) :
  playerAWins rules ↔ ¬(isFibonacci rules.n) := by sorry

end NUMINAMATH_CALUDE_winning_strategy_l2117_211777


namespace NUMINAMATH_CALUDE_min_value_theorem_l2117_211791

theorem min_value_theorem (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 2) :
  ∀ a b c, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 2 → 
    (1/3) * x^3 + y^2 + z ≤ (1/3) * a^3 + b^2 + c :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2117_211791


namespace NUMINAMATH_CALUDE_acid_mixture_problem_l2117_211787

theorem acid_mixture_problem (a w : ℝ) 
  (h1 : a / (a + w + 2) = 1/4)  -- 25% acid after adding 2 oz water
  (h2 : (a + 2) / (a + w + 4) = 2/5)  -- 40% acid after adding 2 oz acid
  : a / (a + w) = 1/3 :=  -- Original mixture is 33 1/3% acid
by sorry

end NUMINAMATH_CALUDE_acid_mixture_problem_l2117_211787


namespace NUMINAMATH_CALUDE_prime_triplet_equation_l2117_211755

theorem prime_triplet_equation (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r →
  (p : ℚ) / q - 4 / (r + 1) = 1 →
  ((p = 7 ∧ q = 3 ∧ r = 2) ∨ (p = 5 ∧ q = 3 ∧ r = 5)) := by
  sorry

end NUMINAMATH_CALUDE_prime_triplet_equation_l2117_211755


namespace NUMINAMATH_CALUDE_tommy_profit_l2117_211713

/-- Represents the types of crates Tommy has --/
inductive CrateType
| A
| B
| C

/-- Represents the types of fruits --/
inductive FruitType
| Tomato
| Orange
| Apple

/-- Calculates Tommy's profit from selling fruits --/
def calculate_profit : ℝ :=
  let crate_a_count : ℕ := 2
  let crate_b_count : ℕ := 3
  let crate_c_count : ℕ := 1

  let crate_a_cost : ℝ := 220
  let crate_b_cost : ℝ := 375
  let crate_c_cost : ℝ := 180

  let transportation_cost : ℝ := 50
  let packing_cost : ℝ := 30

  let crate_a_capacity : FruitType → ℝ := λ f => match f with
    | FruitType.Tomato => 20
    | FruitType.Orange => 10
    | FruitType.Apple => 0

  let crate_b_capacity : FruitType → ℝ := λ f => match f with
    | FruitType.Tomato => 25
    | FruitType.Orange => 15
    | FruitType.Apple => 5

  let crate_c_capacity : FruitType → ℝ := λ f => match f with
    | FruitType.Tomato => 30
    | FruitType.Orange => 0
    | FruitType.Apple => 20

  let rotten_a : FruitType → ℝ := λ f => match f with
    | FruitType.Tomato => 4
    | FruitType.Orange => 2
    | FruitType.Apple => 0

  let rotten_b : FruitType → ℝ := λ f => match f with
    | FruitType.Tomato => 5
    | FruitType.Orange => 3
    | FruitType.Apple => 1

  let rotten_c : FruitType → ℝ := λ f => match f with
    | FruitType.Tomato => 3
    | FruitType.Orange => 0
    | FruitType.Apple => 2

  let selling_price : CrateType → FruitType → ℝ := λ c f => match c, f with
    | CrateType.A, FruitType.Tomato => 5
    | CrateType.A, FruitType.Orange => 4
    | CrateType.B, FruitType.Tomato => 6
    | CrateType.B, FruitType.Orange => 4.5
    | CrateType.B, FruitType.Apple => 3
    | CrateType.C, FruitType.Tomato => 7
    | CrateType.C, FruitType.Apple => 3.5
    | _, _ => 0

  let total_cost : ℝ := crate_a_cost + crate_b_cost + crate_c_cost + transportation_cost + packing_cost

  let revenue_a : ℝ := crate_a_count * (
    (crate_a_capacity FruitType.Tomato - rotten_a FruitType.Tomato) * selling_price CrateType.A FruitType.Tomato +
    (crate_a_capacity FruitType.Orange - rotten_a FruitType.Orange) * selling_price CrateType.A FruitType.Orange
  )

  let revenue_b : ℝ := crate_b_count * (
    (crate_b_capacity FruitType.Tomato - rotten_b FruitType.Tomato) * selling_price CrateType.B FruitType.Tomato +
    (crate_b_capacity FruitType.Orange - rotten_b FruitType.Orange) * selling_price CrateType.B FruitType.Orange +
    (crate_b_capacity FruitType.Apple - rotten_b FruitType.Apple) * selling_price CrateType.B FruitType.Apple
  )

  let revenue_c : ℝ := crate_c_count * (
    (crate_c_capacity FruitType.Tomato - rotten_c FruitType.Tomato) * selling_price CrateType.C FruitType.Tomato +
    (crate_c_capacity FruitType.Apple - rotten_c FruitType.Apple) * selling_price CrateType.C FruitType.Apple
  )

  let total_revenue : ℝ := revenue_a + revenue_b + revenue_c

  total_revenue - total_cost

/-- Theorem stating that Tommy's profit is $179 --/
theorem tommy_profit : calculate_profit = 179 := by sorry

end NUMINAMATH_CALUDE_tommy_profit_l2117_211713


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2117_211779

def A : Set (ℝ × ℝ) := {p | p.2 = 3 * p.1 - 2}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1 ^ 2}

theorem intersection_of_A_and_B :
  A ∩ B = {(1, 1), (2, 4)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2117_211779


namespace NUMINAMATH_CALUDE_divisor_problem_l2117_211734

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 166 → quotient = 9 → remainder = 4 → 
  dividend = divisor * quotient + remainder →
  divisor = 18 := by sorry

end NUMINAMATH_CALUDE_divisor_problem_l2117_211734


namespace NUMINAMATH_CALUDE_terminal_side_equivalence_l2117_211703

/-- Two angles have the same terminal side if their difference is an integer multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α - β = 360 * k

/-- Prove that -330° has the same terminal side as 30° -/
theorem terminal_side_equivalence : same_terminal_side (-330) 30 := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_equivalence_l2117_211703


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l2117_211712

-- Define a circle in 2D plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if two circles intersect
def intersect (c1 c2 : Circle) : Prop := sorry

-- Define a function to get intersection points of two circles
def intersection_points (c1 c2 : Circle) : Set (ℝ × ℝ) := sorry

-- Define a function to check if points are concyclic or collinear
def concyclic_or_collinear (points : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem circle_intersection_theorem (S1 S2 S3 S4 : Circle) :
  intersect S1 S2 ∧ intersect S1 S4 ∧ intersect S3 S2 ∧ intersect S3 S4 →
  concyclic_or_collinear (intersection_points S1 S2 ∪ intersection_points S3 S4) →
  concyclic_or_collinear (intersection_points S1 S4 ∪ intersection_points S2 S3) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l2117_211712


namespace NUMINAMATH_CALUDE_watermelon_price_l2117_211706

theorem watermelon_price : 
  let base_price : ℕ := 5000
  let additional_cost : ℕ := 200
  let total_price : ℕ := base_price + additional_cost
  let price_in_thousands : ℚ := total_price / 1000
  price_in_thousands = 5.2 := by sorry

end NUMINAMATH_CALUDE_watermelon_price_l2117_211706


namespace NUMINAMATH_CALUDE_line_through_points_l2117_211753

/-- Given a line y = ax + b passing through points (3, 7) and (9/2, 13), prove that a - b = 9 -/
theorem line_through_points (a b : ℝ) : 
  (7 = a * 3 + b) → (13 = a * (9/2) + b) → a - b = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2117_211753


namespace NUMINAMATH_CALUDE_supermarket_growth_l2117_211774

/-- Represents the growth of a supermarket's turnover from January to March -/
theorem supermarket_growth (x : ℝ) : 
  (36 : ℝ) * (1 + x)^2 = 48 ↔ 
  (∃ (jan mar : ℝ), 
    jan = 36 ∧ 
    mar = 48 ∧ 
    mar = jan * (1 + x)^2 ∧ 
    x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_supermarket_growth_l2117_211774


namespace NUMINAMATH_CALUDE_prime_square_diff_divisibility_l2117_211744

theorem prime_square_diff_divisibility (p q : ℕ) (k : ℤ) : 
  Prime p → Prime q → p > 5 → q > 5 → p ≠ q → 
  (p^2 : ℤ) - (q^2 : ℤ) = 6 * k → 
  (p^2 : ℤ) - (q^2 : ℤ) ≡ 0 [ZMOD 24] :=
sorry

end NUMINAMATH_CALUDE_prime_square_diff_divisibility_l2117_211744


namespace NUMINAMATH_CALUDE_largest_element_in_S_l2117_211724

def a : ℝ := -4

def S : Set ℝ := { -2 * a^2, 5 * a, 40 / a, 3 * a^2, 2 }

theorem largest_element_in_S : ∀ x ∈ S, x ≤ (3 * a^2) := by
  sorry

end NUMINAMATH_CALUDE_largest_element_in_S_l2117_211724


namespace NUMINAMATH_CALUDE_lcm_gcd_product_24_36_l2117_211776

theorem lcm_gcd_product_24_36 : Nat.lcm 24 36 * Nat.gcd 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_24_36_l2117_211776


namespace NUMINAMATH_CALUDE_product_sum_puzzle_l2117_211749

theorem product_sum_puzzle :
  ∃ (a b c : ℤ), (a * b + c = 40) ∧ (a + b ≠ 18) ∧
  (∃ (a' b' c' : ℤ), (a' * b' + c' = 40) ∧ (a' + b' ≠ 18) ∧ (c' ≠ c)) :=
by sorry

end NUMINAMATH_CALUDE_product_sum_puzzle_l2117_211749


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_two_right_triangle_condition_l2117_211771

def f (a : ℝ) (x : ℝ) := |x + 1| - |a * x - 3|

theorem solution_set_when_a_eq_two :
  {x : ℝ | f 2 x > 1} = {x : ℝ | 1 < x ∧ x < 3} := by sorry

theorem right_triangle_condition (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, f a x = y ∧ f a y = 0 ∧ x ≠ y ∧ (x - y)^2 + (f a x)^2 = (x - y)^2 + y^2) →
  a = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_two_right_triangle_condition_l2117_211771


namespace NUMINAMATH_CALUDE_negation_of_forall_even_square_plus_self_l2117_211793

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

theorem negation_of_forall_even_square_plus_self :
  (¬ ∀ n : ℕ, is_even (n^2 + n)) ↔ (∃ x : ℕ, ¬ is_even (x^2 + x)) :=
sorry

end NUMINAMATH_CALUDE_negation_of_forall_even_square_plus_self_l2117_211793


namespace NUMINAMATH_CALUDE_triangle_and_squares_area_l2117_211732

theorem triangle_and_squares_area (x : ℝ) : 
  let triangle_area := (1/2) * (3*x) * (4*x)
  let square1_area := (3*x)^2
  let square2_area := (4*x)^2
  let square3_area := (6*x)^2
  let total_area := triangle_area + square1_area + square2_area + square3_area
  total_area = 1288 → x = Real.sqrt (1288/67) := by
sorry

end NUMINAMATH_CALUDE_triangle_and_squares_area_l2117_211732


namespace NUMINAMATH_CALUDE_equation_solutions_l2117_211764

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), 
  (3 * x₁^2 + 3 * x₁ + 6 = |(-20 + 5 * x₁)|) ∧ 
  (3 * x₂^2 + 3 * x₂ + 6 = |(-20 + 5 * x₂)|) ∧ 
  (x₁ ≠ x₂) ∧ 
  (-4 < x₁) ∧ (x₁ < 2) ∧ 
  (-4 < x₂) ∧ (x₂ < 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2117_211764


namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_triple_exterior_polygon_sides_proof_l2117_211707

theorem polygon_sides_when_interior_triple_exterior : ℕ → Prop :=
  fun n =>
    (((n : ℝ) - 2) * 180 = 3 * 360) →
    n = 8

-- Proof
theorem polygon_sides_proof : polygon_sides_when_interior_triple_exterior 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_triple_exterior_polygon_sides_proof_l2117_211707


namespace NUMINAMATH_CALUDE_bookstore_shipment_calculation_bookstore_shipment_proof_l2117_211733

/-- Calculates the number of books received in a shipment given initial inventory, sales data, and final inventory. -/
theorem bookstore_shipment_calculation 
  (initial_inventory : ℕ) 
  (saturday_in_store : ℕ) 
  (saturday_online : ℕ) 
  (sunday_in_store_multiplier : ℕ) 
  (sunday_online_increase : ℕ) 
  (final_inventory : ℕ) : ℕ :=
  let total_saturday_sales := saturday_in_store + saturday_online
  let sunday_in_store := sunday_in_store_multiplier * saturday_in_store
  let sunday_online := saturday_online + sunday_online_increase
  let total_sunday_sales := sunday_in_store + sunday_online
  let total_sales := total_saturday_sales + total_sunday_sales
  let inventory_after_sales := initial_inventory - total_sales
  final_inventory - inventory_after_sales

/-- Proves that the bookstore received 160 books in the shipment. -/
theorem bookstore_shipment_proof : 
  bookstore_shipment_calculation 743 37 128 2 34 502 = 160 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_shipment_calculation_bookstore_shipment_proof_l2117_211733


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l2117_211709

theorem rectangle_dimension_change (L W : ℝ) (L' W' : ℝ) : 
  L > 0 ∧ W > 0 →  -- Ensure positive dimensions
  L' = 1.4 * L →   -- Length increased by 40%
  L * W = L' * W' → -- Area remains constant
  (W - W') / W = 0.2857 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l2117_211709


namespace NUMINAMATH_CALUDE_magic_money_box_l2117_211768

def tripleEachDay (initial : ℕ) (days : ℕ) : ℕ :=
  initial * (3 ^ days)

theorem magic_money_box (initial : ℕ) (days : ℕ) 
  (h1 : initial = 5) (h2 : days = 7) : 
  tripleEachDay initial days = 10935 := by
  sorry

end NUMINAMATH_CALUDE_magic_money_box_l2117_211768


namespace NUMINAMATH_CALUDE_tangent_half_identities_l2117_211705

theorem tangent_half_identities (α : Real) (h : Real.tan α = 1/2) :
  ((4 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 2/3) ∧
  (Real.sin α ^ 2 - Real.sin (2 * α) = -3/5) := by
  sorry

end NUMINAMATH_CALUDE_tangent_half_identities_l2117_211705


namespace NUMINAMATH_CALUDE_art_book_cost_is_two_l2117_211718

/-- The cost of each art book given the number of books and their prices --/
def cost_of_art_book (math_books science_books art_books : ℕ) 
                     (total_cost : ℚ) (math_science_cost : ℚ) : ℚ :=
  (total_cost - (math_books + science_books : ℚ) * math_science_cost) / art_books

/-- Theorem stating that the cost of each art book is $2 --/
theorem art_book_cost_is_two :
  cost_of_art_book 2 6 3 30 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_art_book_cost_is_two_l2117_211718


namespace NUMINAMATH_CALUDE_two_distinct_roots_l2117_211761

-- Define the function representing the equation
def f (x p : ℝ) : ℝ := x^2 - 2*|x| - p

-- State the theorem
theorem two_distinct_roots (p : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f x p = 0 ∧ f y p = 0) ↔ p > -1 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l2117_211761


namespace NUMINAMATH_CALUDE_hash_five_three_l2117_211783

-- Define the # operation
def hash (a b : ℤ) : ℤ := 4 * a + 6 * b

-- Theorem statement
theorem hash_five_three : hash 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_hash_five_three_l2117_211783


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2117_211790

theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + y^2 = 3/4}
  let tangent_line := {(x, y) : ℝ × ℝ | ∃ k, y = k * x ∧ k^2 = 3}
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let e := Real.sqrt (1 + b^2 / a^2)
  (∃ p q : ℝ × ℝ, p ∈ tangent_line ∧ q ∈ tangent_line ∧ p ∈ hyperbola ∧ q ∈ hyperbola ∧ p ≠ q) →
  e > 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2117_211790


namespace NUMINAMATH_CALUDE_complex_square_simplification_l2117_211726

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (5 - 3 * i)^2 = 16 - 30 * i := by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l2117_211726


namespace NUMINAMATH_CALUDE_cone_volume_from_semicircle_l2117_211775

theorem cone_volume_from_semicircle (r : ℝ) (h : r = 6) :
  let l := r  -- slant height
  let base_radius := r / 2  -- derived from circumference equality
  let height := Real.sqrt (l^2 - base_radius^2)
  let volume := (1/3) * Real.pi * base_radius^2 * height
  volume = 9 * Real.sqrt 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_from_semicircle_l2117_211775


namespace NUMINAMATH_CALUDE_tangent_line_equality_l2117_211711

theorem tangent_line_equality (x₁ x₂ y₁ y₂ : ℝ) :
  (∃ m b : ℝ, (∀ x : ℝ, y₁ + (Real.exp x₁) * (x - x₁) = m * x + b) ∧
              (∀ x : ℝ, y₂ + (1 / x₂) * (x - x₂) = m * x + b)) →
  (x₁ + 1) * (x₂ - 1) = -2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equality_l2117_211711


namespace NUMINAMATH_CALUDE_supplementary_angles_problem_l2117_211794

theorem supplementary_angles_problem (x y : ℝ) : 
  x + y = 180 → 
  y = x + 18 → 
  y = 99 := by
sorry

end NUMINAMATH_CALUDE_supplementary_angles_problem_l2117_211794


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2117_211739

theorem linear_equation_solution (x y : ℝ) :
  2 * x + y - 5 = 0 → x = (5 - y) / 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2117_211739


namespace NUMINAMATH_CALUDE_birds_left_l2117_211798

theorem birds_left (initial_chickens ducks turkeys chickens_sold : ℕ) :
  initial_chickens ≥ chickens_sold →
  (initial_chickens - chickens_sold + ducks + turkeys : ℕ) =
    initial_chickens + ducks + turkeys - chickens_sold :=
by sorry

end NUMINAMATH_CALUDE_birds_left_l2117_211798


namespace NUMINAMATH_CALUDE_inequality_proof_l2117_211729

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2117_211729


namespace NUMINAMATH_CALUDE_complex_roots_circle_radius_l2117_211786

theorem complex_roots_circle_radius : 
  ∀ z : ℂ, (z - 2)^6 = 64 * z^6 → Complex.abs (z - (2/3 : ℂ)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_circle_radius_l2117_211786


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l2117_211710

theorem cubic_expression_evaluation : (3^3 - 3) - (4^3 - 4) + (5^3 - 5) = 84 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l2117_211710


namespace NUMINAMATH_CALUDE_first_term_arithmetic_progression_l2117_211756

/-- 
For a decreasing arithmetic progression with first term a, sum S, 
number of terms n, and common difference d, the following equation holds:
a = S/n + (n-1)d/2
-/
theorem first_term_arithmetic_progression 
  (a : ℝ) (S : ℝ) (n : ℕ) (d : ℝ) 
  (h1 : n > 0) 
  (h2 : d < 0) -- Ensures it's a decreasing progression
  (h3 : S = n/2 * (2*a + (n-1)*d)) -- Sum formula for arithmetic progression
  : a = S/n + (n-1)*d/2 := by
  sorry

end NUMINAMATH_CALUDE_first_term_arithmetic_progression_l2117_211756


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2117_211737

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = (x^4 + x^2 + 3) * (2*x^5 + x^3 + 7)) ∧ 
    (f 0 = 21) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2117_211737


namespace NUMINAMATH_CALUDE_log_sine_absolute_value_sum_l2117_211720

theorem log_sine_absolute_value_sum (x : ℝ) (θ : ℝ) 
  (h : Real.log x / Real.log 2 = 2 + Real.sin θ) : 
  |x + 1| + |x - 10| = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_sine_absolute_value_sum_l2117_211720


namespace NUMINAMATH_CALUDE_shaded_area_approx_l2117_211742

/-- The area of a 4 x 6 rectangle minus a circle with diameter 2 is approximately 21 -/
theorem shaded_area_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  (4 * 6 : ℝ) - Real.pi * (2 / 2)^2 = 21 + ε := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_approx_l2117_211742


namespace NUMINAMATH_CALUDE_tan_105_degrees_l2117_211781

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l2117_211781


namespace NUMINAMATH_CALUDE_min_bounces_to_height_ball_bounce_problem_l2117_211725

def bounce_height (initial_height : ℝ) (bounce_ratio : ℝ) (n : ℕ) : ℝ :=
  initial_height * (bounce_ratio ^ n)

theorem min_bounces_to_height (initial_height bounce_ratio target_height : ℝ) :
  ∃ (n : ℕ), 
    (∀ (k : ℕ), k < n → bounce_height initial_height bounce_ratio k ≥ target_height) ∧
    bounce_height initial_height bounce_ratio n < target_height :=
  sorry

theorem ball_bounce_problem :
  let initial_height := 243
  let bounce_ratio := 2/3
  let target_height := 30
  ∃ (n : ℕ), n = 6 ∧
    (∀ (k : ℕ), k < n → bounce_height initial_height bounce_ratio k ≥ target_height) ∧
    bounce_height initial_height bounce_ratio n < target_height :=
  sorry

end NUMINAMATH_CALUDE_min_bounces_to_height_ball_bounce_problem_l2117_211725


namespace NUMINAMATH_CALUDE_f_geq_4_iff_valid_a_range_f_3_geq_4_l2117_211767

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 3/a| + |x - a|

def valid_a_range (a : ℝ) : Prop :=
  a ∈ Set.Iic (-3) ∪ Set.Icc (-1) 0 ∪ Set.Ioc 0 1 ∪ Set.Ici 3

theorem f_geq_4_iff_valid_a_range (a : ℝ) (h : a ≠ 0) :
  (∀ x, f a x ≥ 4) ↔ valid_a_range a := by sorry

theorem f_3_geq_4 (a : ℝ) (h : a ≠ 0) : f a 3 ≥ 4 := by sorry

end NUMINAMATH_CALUDE_f_geq_4_iff_valid_a_range_f_3_geq_4_l2117_211767


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2117_211728

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(38.2 : ℝ)⌉ = 35 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2117_211728


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_a_equals_two_l2117_211702

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Given two vectors m and n, if they are parallel, then the second component of n is 2 -/
theorem parallel_vectors_imply_a_equals_two (m n : ℝ × ℝ) 
    (hm : m = (2, 1)) 
    (hn : ∃ a : ℝ, n = (4, a)) 
    (h_parallel : are_parallel m n) : 
    n.2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_a_equals_two_l2117_211702


namespace NUMINAMATH_CALUDE_sues_shoe_probability_l2117_211778

/-- Represents the number of pairs of shoes for each color --/
structure ShoePairs where
  black : ℕ
  brown : ℕ
  gray : ℕ

/-- Calculates the probability of selecting two shoes of the same color,
    one left and one right, given the number of pairs for each color --/
def samePairColorProbability (pairs : ShoePairs) : ℚ :=
  let totalShoes := 2 * (pairs.black + pairs.brown + pairs.gray)
  let blackProb := (2 * pairs.black) * pairs.black / (totalShoes * (totalShoes - 1))
  let brownProb := (2 * pairs.brown) * pairs.brown / (totalShoes * (totalShoes - 1))
  let grayProb := (2 * pairs.gray) * pairs.gray / (totalShoes * (totalShoes - 1))
  blackProb + brownProb + grayProb

/-- Theorem stating that for Sue's shoe collection, the probability of
    selecting two shoes of the same color, one left and one right, is 7/33 --/
theorem sues_shoe_probability :
  samePairColorProbability ⟨6, 3, 2⟩ = 7 / 33 := by
  sorry

end NUMINAMATH_CALUDE_sues_shoe_probability_l2117_211778


namespace NUMINAMATH_CALUDE_semi_annual_compound_interest_rate_l2117_211716

/-- Proves that the annual interest rate of a semi-annually compounded account is approximately 7.96%
    given specific conditions on the initial investment and interest earned. -/
theorem semi_annual_compound_interest_rate (principal : ℝ) (simple_rate : ℝ) (diff : ℝ) :
  principal = 5000 →
  simple_rate = 0.08 →
  diff = 6 →
  ∃ (compound_rate : ℝ),
    (principal * (1 + compound_rate / 2)^2 - principal) = 
    (principal * simple_rate + diff) ∧
    abs (compound_rate - 0.0796) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_semi_annual_compound_interest_rate_l2117_211716


namespace NUMINAMATH_CALUDE_book_distribution_l2117_211773

theorem book_distribution (people : ℕ) (books : ℕ) : 
  (5 * people = books + 2) →
  (4 * people + 3 = books) →
  (people = 5 ∧ books = 23) := by
sorry

end NUMINAMATH_CALUDE_book_distribution_l2117_211773


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2117_211727

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude : ℝ) 
  (base : ℝ) 
  (h1 : area = 450) 
  (h2 : altitude = 2 * base) 
  (h3 : area = base * altitude) : 
  base = 15 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2117_211727


namespace NUMINAMATH_CALUDE_specific_event_handshakes_l2117_211708

/-- Represents a social event with two groups of people -/
structure SocialEvent where
  total_people : ℕ
  group_a_size : ℕ
  group_b_size : ℕ
  h_total : total_people = group_a_size + group_b_size
  h_group_a : group_a_size > 0
  h_group_b : group_b_size > 0

/-- Calculates the number of handshakes in a social event -/
def handshakes (event : SocialEvent) : ℕ :=
  event.group_a_size * event.group_b_size

/-- Theorem stating the number of handshakes in the specific social event -/
theorem specific_event_handshakes :
  ∃ (event : SocialEvent),
    event.total_people = 40 ∧
    event.group_a_size = 25 ∧
    event.group_b_size = 15 ∧
    handshakes event = 375 := by
  sorry

end NUMINAMATH_CALUDE_specific_event_handshakes_l2117_211708


namespace NUMINAMATH_CALUDE_hancho_height_calculation_l2117_211743

/-- Hancho's height in centimeters, given Hansol's height and the ratio between their heights -/
def hanchos_height (hansols_height : ℝ) (height_ratio : ℝ) : ℝ :=
  hansols_height * height_ratio

/-- Theorem stating that Hancho's height is 142.57 cm -/
theorem hancho_height_calculation :
  let hansols_height : ℝ := 134.5
  let height_ratio : ℝ := 1.06
  hanchos_height hansols_height height_ratio = 142.57 := by sorry

end NUMINAMATH_CALUDE_hancho_height_calculation_l2117_211743


namespace NUMINAMATH_CALUDE_average_difference_l2117_211730

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + 80 + x) / 3) + 5 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l2117_211730


namespace NUMINAMATH_CALUDE_no_meetings_before_return_l2117_211765

/-- The number of times two boys meet on a circular track before returning to their starting point -/
def number_of_meetings (circumference : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℕ :=
  sorry

theorem no_meetings_before_return :
  let circumference : ℝ := 120
  let speed1 : ℝ := 6
  let speed2 : ℝ := 10
  number_of_meetings circumference speed1 speed2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_meetings_before_return_l2117_211765


namespace NUMINAMATH_CALUDE_jesse_blocks_l2117_211750

theorem jesse_blocks (cityscape farmhouse zoo fence1 fence2 fence3 left : ℕ) 
  (h1 : cityscape = 80)
  (h2 : farmhouse = 123)
  (h3 : zoo = 95)
  (h4 : fence1 = 57)
  (h5 : fence2 = 43)
  (h6 : fence3 = 62)
  (h7 : left = 84) :
  cityscape + farmhouse + zoo + fence1 + fence2 + fence3 + left = 544 := by
  sorry

end NUMINAMATH_CALUDE_jesse_blocks_l2117_211750


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2117_211785

theorem pie_eating_contest (first_round first_second_round second_total : ℚ) 
  (h1 : first_round = 5/6)
  (h2 : first_second_round = 1/6)
  (h3 : second_total = 2/3) :
  first_round + first_second_round - second_total = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2117_211785


namespace NUMINAMATH_CALUDE_martha_apples_l2117_211789

/-- Given Martha's initial apples and the distribution to her friends, 
    prove the number of additional apples she needs to give away to be left with 4. -/
theorem martha_apples (initial_apples : ℕ) (jane_apples : ℕ) (james_extra : ℕ) :
  initial_apples = 20 →
  jane_apples = 5 →
  james_extra = 2 →
  initial_apples - jane_apples - (jane_apples + james_extra) - 4 = 4 :=
by sorry

end NUMINAMATH_CALUDE_martha_apples_l2117_211789


namespace NUMINAMATH_CALUDE_parabola_shift_l2117_211797

def original_parabola (x : ℝ) : ℝ := -2 * x^2 + 4

def shifted_parabola (x : ℝ) : ℝ := -2 * (x + 2)^2 + 7

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 2) + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l2117_211797


namespace NUMINAMATH_CALUDE_remainder_proof_l2117_211735

theorem remainder_proof (n : ℕ) (h1 : n = 88) (h2 : (3815 - 31) % n = 0) (h3 : ∃ r, (4521 - r) % n = 0) :
  4521 % n = 33 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l2117_211735


namespace NUMINAMATH_CALUDE_cut_cube_height_l2117_211759

/-- The height of a cube with a corner cut off -/
theorem cut_cube_height : 
  let s : ℝ := 2  -- side length of the original cube
  let triangle_side : ℝ := s * Real.sqrt 2  -- side length of the cut triangle
  let base_area : ℝ := (Real.sqrt 3 / 4) * triangle_side ^ 2  -- area of the cut face
  let pyramid_volume : ℝ := s ^ 3 / 6  -- volume of the cut-off pyramid
  let h : ℝ := pyramid_volume / (base_area / 6)  -- height of the cut-off pyramid
  2 - h = 2 - (2 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_cut_cube_height_l2117_211759


namespace NUMINAMATH_CALUDE_max_n_given_average_l2117_211738

theorem max_n_given_average (m n : ℕ+) : 
  (m + n : ℚ) / 2 = 5 → n ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_n_given_average_l2117_211738
