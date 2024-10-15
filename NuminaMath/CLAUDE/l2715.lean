import Mathlib

namespace NUMINAMATH_CALUDE_pumpkin_weight_problem_l2715_271530

/-- Given two pumpkins with a total weight of 12.7 pounds, 
    if one pumpkin weighs 4 pounds, then the other pumpkin weighs 8.7 pounds. -/
theorem pumpkin_weight_problem (total_weight : ℝ) (pumpkin1_weight : ℝ) (pumpkin2_weight : ℝ) :
  total_weight = 12.7 →
  pumpkin1_weight = 4 →
  total_weight = pumpkin1_weight + pumpkin2_weight →
  pumpkin2_weight = 8.7 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_weight_problem_l2715_271530


namespace NUMINAMATH_CALUDE_study_supplies_cost_l2715_271566

/-- The cost of study supplies -/
theorem study_supplies_cost 
  (x y z : ℚ) -- x: cost of a pencil, y: cost of an exercise book, z: cost of a ballpoint pen
  (h1 : 3*x + 7*y + z = 3.15) -- First condition
  (h2 : 4*x + 10*y + z = 4.2) -- Second condition
  : x + y + z = 1.05 := by sorry

end NUMINAMATH_CALUDE_study_supplies_cost_l2715_271566


namespace NUMINAMATH_CALUDE_tangent_line_at_one_unique_solution_l2715_271547

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a - 1/x - Real.log x

-- Part 1: Tangent line when a = 2
theorem tangent_line_at_one (x : ℝ) :
  let a : ℝ := 2
  let f' : ℝ → ℝ := λ x => (1 - x) / (x^2)
  (f' 1 = 0) ∧ (f a 1 = 1) → (λ y => y = 1) = (λ y => y = f' 1 * (x - 1) + f a 1) :=
sorry

-- Part 2: Unique solution when a = 1
theorem unique_solution :
  (∃! x : ℝ, f 1 x = 0) ∧ (∀ a : ℝ, a ≠ 1 → ¬(∃! x : ℝ, f a x = 0)) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_one_unique_solution_l2715_271547


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l2715_271540

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (prod_sum_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c + 2*(a + b + c) = 672 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l2715_271540


namespace NUMINAMATH_CALUDE_divisible_number_is_six_l2715_271533

/-- The number of three-digit numbers divisible by the specific number -/
def divisible_count : ℕ := 150

/-- The lower bound of three-digit numbers -/
def lower_bound : ℕ := 100

/-- The upper bound of three-digit numbers -/
def upper_bound : ℕ := 999

/-- The total count of three-digit numbers -/
def total_count : ℕ := upper_bound - lower_bound + 1

theorem divisible_number_is_six :
  ∃ (n : ℕ), n = 6 ∧
  (∀ k : ℕ, lower_bound ≤ k ∧ k ≤ upper_bound →
    (divisible_count * n = total_count)) :=
sorry

end NUMINAMATH_CALUDE_divisible_number_is_six_l2715_271533


namespace NUMINAMATH_CALUDE_total_staff_is_250_l2715_271569

/-- Represents a hospital with doctors and nurses -/
structure Hospital where
  doctors : ℕ
  nurses : ℕ

/-- The total number of staff (doctors and nurses) in a hospital -/
def Hospital.total (h : Hospital) : ℕ := h.doctors + h.nurses

/-- A hospital satisfying the given conditions -/
def special_hospital : Hospital :=
  { doctors := 100,  -- This is derived from the ratio, not given directly
    nurses := 150 }

theorem total_staff_is_250 :
  (special_hospital.doctors : ℚ) / special_hospital.nurses = 2 / 3 ∧
  special_hospital.nurses = 150 →
  special_hospital.total = 250 := by
  sorry

end NUMINAMATH_CALUDE_total_staff_is_250_l2715_271569


namespace NUMINAMATH_CALUDE_symmetric_probability_l2715_271534

/-- Represents a standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The number of dice rolled -/
def numDice : Nat := 8

/-- The sum we're comparing to -/
def givenSum : Nat := 15

/-- The sum we're proving has the same probability -/
def symmetricSum : Nat := 41

/-- Function to calculate the probability of a specific sum when rolling n dice -/
noncomputable def probability (n : Nat) (sum : Nat) : Real := sorry

theorem symmetric_probability : 
  probability numDice givenSum = probability numDice symmetricSum := by sorry

end NUMINAMATH_CALUDE_symmetric_probability_l2715_271534


namespace NUMINAMATH_CALUDE_legos_lost_l2715_271588

def initial_legos : ℕ := 2080
def current_legos : ℕ := 2063

theorem legos_lost : initial_legos - current_legos = 17 := by
  sorry

end NUMINAMATH_CALUDE_legos_lost_l2715_271588


namespace NUMINAMATH_CALUDE_gift_exchange_equation_l2715_271574

/-- Represents a gathering of people exchanging gifts -/
structure Gathering where
  /-- The number of attendees -/
  attendees : ℕ
  /-- The total number of gifts exchanged -/
  gifts : ℕ
  /-- Each pair of attendees exchanges a different small gift -/
  unique_exchanges : ∀ (a b : Fin attendees), a ≠ b → True

/-- The theorem stating the relationship between attendees and gifts exchanged -/
theorem gift_exchange_equation (g : Gathering) (h : g.gifts = 56) :
  g.attendees * (g.attendees - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_gift_exchange_equation_l2715_271574


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2715_271542

/-- A sequence where each term is 1/3 of the previous term -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = (1 / 3) * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h : geometric_sequence a) (h1 : a 4 + a 5 = 4) : 
  a 2 + a 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2715_271542


namespace NUMINAMATH_CALUDE_stop_duration_l2715_271548

/-- Calculates the duration of a stop given the total distance, speed, and total travel time. -/
theorem stop_duration (distance : ℝ) (speed : ℝ) (total_time : ℝ) 
  (h1 : distance = 360) 
  (h2 : speed = 60) 
  (h3 : total_time = 7) :
  total_time - distance / speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_stop_duration_l2715_271548


namespace NUMINAMATH_CALUDE_triangle_area_l2715_271538

/-- Given a triangle with perimeter 32 and inradius 2.5, its area is 40 -/
theorem triangle_area (p r a : ℝ) (h1 : p = 32) (h2 : r = 2.5) (h3 : a = p * r / 4) : a = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2715_271538


namespace NUMINAMATH_CALUDE_max_tan_alpha_l2715_271564

theorem max_tan_alpha (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.tan (α + β) = 9 * Real.tan β) : 
  ∃ (max_tan_α : Real), max_tan_α = 4/3 ∧ 
    ∀ (γ : Real), (0 < γ ∧ γ < π/2 ∧ ∃ (δ : Real), (0 < δ ∧ δ < π/2 ∧ Real.tan (γ + δ) = 9 * Real.tan δ)) 
      → Real.tan γ ≤ max_tan_α :=
sorry

end NUMINAMATH_CALUDE_max_tan_alpha_l2715_271564


namespace NUMINAMATH_CALUDE_sum_of_two_primes_odd_implies_one_is_two_l2715_271549

theorem sum_of_two_primes_odd_implies_one_is_two (p q : ℕ) :
  Prime p → Prime q → Odd (p + q) → (p = 2 ∨ q = 2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_primes_odd_implies_one_is_two_l2715_271549


namespace NUMINAMATH_CALUDE_circle_line_intersection_range_l2715_271584

/-- Given a circle C and a line l, if they have a common point, 
    then the range of values for a is [-1/2, 1/2) -/
theorem circle_line_intersection_range (a : ℝ) : 
  let C := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*a*x + 2*a*y + 2*a^2 + 2*a - 1 = 0}
  let l := {(x, y) : ℝ × ℝ | x - y - 1 = 0}
  (∃ p, p ∈ C ∩ l) → a ∈ Set.Icc (-1/2) (1/2) := by
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_range_l2715_271584


namespace NUMINAMATH_CALUDE_decimal_88_to_base5_base5_323_to_decimal_decimal_88_equals_base5_323_l2715_271525

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Converts a list of digits in base-5 to its decimal representation -/
def fromBase5 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 5 * acc) 0

theorem decimal_88_to_base5 :
  toBase5 88 = [3, 2, 3] :=
sorry

theorem base5_323_to_decimal :
  fromBase5 [3, 2, 3] = 88 :=
sorry

/-- The base-5 representation of 88 is 323 -/
theorem decimal_88_equals_base5_323 :
  toBase5 88 = [3, 2, 3] ∧ fromBase5 [3, 2, 3] = 88 :=
sorry

end NUMINAMATH_CALUDE_decimal_88_to_base5_base5_323_to_decimal_decimal_88_equals_base5_323_l2715_271525


namespace NUMINAMATH_CALUDE_corner_cut_length_l2715_271544

/-- Given a rectangular sheet of dimensions 48 m x 36 m, if squares of side length x
    are cut from each corner to form an open box with volume 5120 m³,
    then x = 8 m. -/
theorem corner_cut_length (x : ℝ) : 
  x > 0 ∧ x < 18 ∧ (48 - 2*x) * (36 - 2*x) * x = 5120 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_corner_cut_length_l2715_271544


namespace NUMINAMATH_CALUDE_bottles_taken_home_l2715_271595

def bottles_brought : ℕ := 50
def bottles_drunk : ℕ := 38

theorem bottles_taken_home : 
  bottles_brought - bottles_drunk = 12 := by sorry

end NUMINAMATH_CALUDE_bottles_taken_home_l2715_271595


namespace NUMINAMATH_CALUDE_pizza_price_correct_l2715_271590

/-- The price of one box of pizza -/
def pizza_price : ℝ := 12

/-- The price of one pack of potato fries -/
def fries_price : ℝ := 0.3

/-- The price of one can of soda -/
def soda_price : ℝ := 2

/-- The number of pizza boxes sold -/
def pizza_sold : ℕ := 15

/-- The number of potato fries packs sold -/
def fries_sold : ℕ := 40

/-- The number of soda cans sold -/
def soda_sold : ℕ := 25

/-- The fundraising goal -/
def goal : ℝ := 500

/-- The amount still needed to reach the goal -/
def amount_needed : ℝ := 258

theorem pizza_price_correct : 
  pizza_price * pizza_sold + fries_price * fries_sold + soda_price * soda_sold = goal - amount_needed :=
by sorry

end NUMINAMATH_CALUDE_pizza_price_correct_l2715_271590


namespace NUMINAMATH_CALUDE_vera_doll_count_l2715_271532

theorem vera_doll_count (aida sophie vera : ℕ) : 
  aida = 2 * sophie →
  sophie = 2 * vera →
  aida + sophie + vera = 140 →
  vera = 20 :=
by sorry

end NUMINAMATH_CALUDE_vera_doll_count_l2715_271532


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2715_271567

theorem complex_magnitude_problem (m : ℝ) : 
  (Complex.I * ((1 + m * Complex.I) * (3 + Complex.I))).re = 0 →
  Complex.abs ((m + 3 * Complex.I) / (1 - Complex.I)) = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2715_271567


namespace NUMINAMATH_CALUDE_proportional_relation_l2715_271586

/-- Given that x is directly proportional to y^4 and y is inversely proportional to z^(1/3),
    prove that if x = 8 when z = 27, then x = 81/32 when z = 64 -/
theorem proportional_relation (x y z : ℝ) (k₁ k₂ : ℝ) 
    (h1 : x = k₁ * y^4)
    (h2 : y = k₂ / z^(1/3))
    (h3 : x = 8 ∧ z = 27) :
    z = 64 → x = 81/32 := by
  sorry

end NUMINAMATH_CALUDE_proportional_relation_l2715_271586


namespace NUMINAMATH_CALUDE_triangle_area_l2715_271513

theorem triangle_area (a b c : ℝ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25) :
  (1/2) * a * b = 84 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2715_271513


namespace NUMINAMATH_CALUDE_petes_number_l2715_271572

theorem petes_number (x : ℝ) : 3 * (2 * x + 12) = 90 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l2715_271572


namespace NUMINAMATH_CALUDE_amaya_total_marks_l2715_271570

/-- Represents the marks scored in different subjects -/
structure Marks where
  music : ℕ
  social_studies : ℕ
  arts : ℕ
  maths : ℕ

/-- Calculates the total marks across all subjects -/
def total_marks (m : Marks) : ℕ :=
  m.music + m.social_studies + m.arts + m.maths

/-- Theorem stating the total marks Amaya scored -/
theorem amaya_total_marks :
  ∀ (m : Marks),
  m.music = 70 →
  m.social_studies = m.music + 10 →
  m.arts - m.maths = 20 →
  m.maths = (9 : ℕ) * m.arts / 10 →
  total_marks m = 530 := by
  sorry


end NUMINAMATH_CALUDE_amaya_total_marks_l2715_271570


namespace NUMINAMATH_CALUDE_inverse_g_75_l2715_271553

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 - 6

-- State the theorem
theorem inverse_g_75 : g⁻¹ 75 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_75_l2715_271553


namespace NUMINAMATH_CALUDE_inheritance_tax_problem_l2715_271509

theorem inheritance_tax_problem (x : ℝ) : 
  0.25 * x + 0.12 * (0.75 * x) = 13600 → x = 40000 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_tax_problem_l2715_271509


namespace NUMINAMATH_CALUDE_lisa_quiz_goal_l2715_271597

theorem lisa_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (completed_as : ℕ) 
  (h1 : total_quizzes = 40)
  (h2 : goal_percentage = 9/10)
  (h3 : completed_quizzes = 25)
  (h4 : completed_as = 20) : 
  (total_quizzes - completed_quizzes : ℤ) - 
  (↑(total_quizzes * goal_percentage.num) / goal_percentage.den - completed_as : ℚ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_lisa_quiz_goal_l2715_271597


namespace NUMINAMATH_CALUDE_bob_arrival_probability_bob_arrival_probability_value_l2715_271592

/-- The probability that Bob arrived before 3:45 PM given that Alice arrived after him,
    when both arrive randomly between 3:00 PM and 4:00 PM. -/
theorem bob_arrival_probability : ℝ :=
  let total_time := 60 -- minutes
  let bob_early_time := 45 -- minutes
  let total_area := (total_time ^ 2) / 2 -- area where Alice arrives after Bob
  let early_area := (bob_early_time ^ 2) / 2 -- area where Bob is early and Alice is after
  early_area / total_area

/-- The probability is equal to 9/16 -/
theorem bob_arrival_probability_value : bob_arrival_probability = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_bob_arrival_probability_bob_arrival_probability_value_l2715_271592


namespace NUMINAMATH_CALUDE_range_of_a_l2715_271518

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + 1 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → a^2 - 3*a - x + 1 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(p a ∧ q a)) ∧ (¬¬(q a)) → a ∈ Set.Ico 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2715_271518


namespace NUMINAMATH_CALUDE_union_of_A_and_I_minus_B_l2715_271583

def I : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 1, 2}

theorem union_of_A_and_I_minus_B : A ∪ (I \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_I_minus_B_l2715_271583


namespace NUMINAMATH_CALUDE_tax_problem_l2715_271576

/-- Proves that the monthly gross income is 127,500 HUF when the tax equals 30% of the annual income --/
theorem tax_problem (x : ℝ) (h1 : x > 1050000) : 
  (267000 + 0.4 * (x - 1050000) = 0.3 * x) → (x / 12 = 127500) := by
  sorry

end NUMINAMATH_CALUDE_tax_problem_l2715_271576


namespace NUMINAMATH_CALUDE_red_light_probability_l2715_271504

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of seeing a red light -/
def probabilityRedLight (d : TrafficLightDuration) : ℚ :=
  d.red / (d.red + d.yellow + d.green)

/-- Theorem: The probability of seeing a red light is 2/5 for the given durations -/
theorem red_light_probability (d : TrafficLightDuration) 
  (h1 : d.red = 30) 
  (h2 : d.yellow = 5) 
  (h3 : d.green = 40) : 
  probabilityRedLight d = 2/5 := by
  sorry

#eval probabilityRedLight ⟨30, 5, 40⟩

end NUMINAMATH_CALUDE_red_light_probability_l2715_271504


namespace NUMINAMATH_CALUDE_equation_solution_l2715_271556

theorem equation_solution : ∃! x : ℝ, 4 * x - 3 = 5 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2715_271556


namespace NUMINAMATH_CALUDE_cosine_sine_values_l2715_271565

/-- If the sum of cos²ⁿ(θ) from n=0 to infinity equals 9, 
    then cos(2θ) = 7/9 and sin²(θ) = 1/9 -/
theorem cosine_sine_values (θ : ℝ) 
  (h : ∑' n, (Real.cos θ)^(2*n) = 9) : 
  Real.cos (2*θ) = 7/9 ∧ Real.sin θ^2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_values_l2715_271565


namespace NUMINAMATH_CALUDE_inequality_proof_l2715_271575

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a + c)^2 + (b + d)^2) ≤ Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) ∧
  Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) ≤ Real.sqrt ((a + c)^2 + (b + d)^2) + (2 * |a * d - b * c|) / Real.sqrt ((a + c)^2 + (b + d)^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2715_271575


namespace NUMINAMATH_CALUDE_area_of_stacked_squares_l2715_271585

/-- The area of a 24-sided polygon formed by stacking three identical square sheets -/
theorem area_of_stacked_squares (side_length : ℝ) (h : side_length = 8) :
  let diagonal := side_length * Real.sqrt 2
  let radius := diagonal / 2
  let triangle_area := (1/2) * radius^2 * Real.sin (π/6)
  let total_area := 12 * triangle_area
  total_area = 96 := by sorry

end NUMINAMATH_CALUDE_area_of_stacked_squares_l2715_271585


namespace NUMINAMATH_CALUDE_mango_purchase_amount_l2715_271550

-- Define the variables
def grapes_kg : ℕ := 3
def grapes_rate : ℕ := 70
def mango_rate : ℕ := 55
def total_paid : ℕ := 705

-- Define the theorem
theorem mango_purchase_amount :
  ∃ (m : ℕ), 
    grapes_kg * grapes_rate + m * mango_rate = total_paid ∧ 
    m = 9 :=
by sorry

end NUMINAMATH_CALUDE_mango_purchase_amount_l2715_271550


namespace NUMINAMATH_CALUDE_train_time_calculation_l2715_271580

/-- Proves that the additional time for train-related activities is 15.5 minutes --/
theorem train_time_calculation (distance : ℝ) (walk_speed : ℝ) (train_speed : ℝ) 
  (walk_time_difference : ℝ) :
  distance = 1.5 →
  walk_speed = 3 →
  train_speed = 20 →
  walk_time_difference = 10 →
  ∃ (x : ℝ), x = 15.5 ∧ 
    (distance / walk_speed) * 60 = (distance / train_speed) * 60 + x + walk_time_difference :=
by
  sorry

end NUMINAMATH_CALUDE_train_time_calculation_l2715_271580


namespace NUMINAMATH_CALUDE_gcd_1963_1891_l2715_271507

theorem gcd_1963_1891 : Nat.gcd 1963 1891 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1963_1891_l2715_271507


namespace NUMINAMATH_CALUDE_even_sequence_sum_l2715_271581

theorem even_sequence_sum (n : ℕ) (sum : ℕ) : sum = n * (n + 1) → 2 * n = 30 :=
  sorry

#check even_sequence_sum

end NUMINAMATH_CALUDE_even_sequence_sum_l2715_271581


namespace NUMINAMATH_CALUDE_massager_usage_time_l2715_271593

/-- The number of vibrations per second at the lowest setting -/
def lowest_vibrations_per_second : ℕ := 1600

/-- The percentage increase in vibrations at the highest setting -/
def highest_setting_increase : ℚ := 60 / 100

/-- The total number of vibrations experienced -/
def total_vibrations : ℕ := 768000

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Calculates the number of minutes Matt uses the massager at the highest setting -/
def usage_time_minutes : ℚ :=
  let highest_vibrations_per_second : ℚ := lowest_vibrations_per_second * (1 + highest_setting_increase)
  let usage_time_seconds : ℚ := total_vibrations / highest_vibrations_per_second
  usage_time_seconds / seconds_per_minute

theorem massager_usage_time :
  usage_time_minutes = 5 := by sorry

end NUMINAMATH_CALUDE_massager_usage_time_l2715_271593


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l2715_271599

/-- The quadratic equation x^2 + 3x - 1 = 0 -/
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x - 1 = 0

/-- The two roots of the quadratic equation -/
noncomputable def root1 : ℝ := sorry
noncomputable def root2 : ℝ := sorry

/-- Proposition p: The two roots have opposite signs -/
def p : Prop := root1 * root2 < 0

/-- Proposition q: The sum of the two roots is 3 -/
def q : Prop := root1 + root2 = 3

theorem quadratic_roots_properties : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l2715_271599


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_l2715_271591

/-- An equilateral hexagon with specified angle measures and area -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side : ℝ
  -- Assertion that the hexagon is equilateral
  is_equilateral : True
  -- Assertion about the interior angles
  angle_condition : True
  -- Area of the hexagon
  area : ℝ
  -- The area is 12
  area_is_twelve : area = 12

/-- The perimeter of a SpecialHexagon is 12 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : h.side * 6 = 12 := by
  sorry

#check special_hexagon_perimeter

end NUMINAMATH_CALUDE_special_hexagon_perimeter_l2715_271591


namespace NUMINAMATH_CALUDE_min_value_fraction_l2715_271501

theorem min_value_fraction (x : ℝ) (h : x > 6) :
  x^2 / (x - 6) ≥ 24 ∧ (x^2 / (x - 6) = 24 ↔ x = 12) := by
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2715_271501


namespace NUMINAMATH_CALUDE_max_ab_value_l2715_271537

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let f := fun x : ℝ => 4 * x^3 - a * x^2 - 2 * b * x + 2
  (∃ (ε : ℝ), ∀ (h : ℝ), 0 < |h| → |h| < ε → f (1 + h) ≤ f 1) →
  (∀ c : ℝ, a * b ≤ c → c ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l2715_271537


namespace NUMINAMATH_CALUDE_percentage_difference_l2715_271555

theorem percentage_difference (x y : ℝ) (h : y = x * (1 + 0.53846153846153854)) :
  x = y * (1 - 0.35) := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l2715_271555


namespace NUMINAMATH_CALUDE_max_value_expression_l2715_271514

theorem max_value_expression (x₁ x₂ x₃ x₄ : ℝ)
  (h₁ : 0 ≤ x₁ ∧ x₁ ≤ 1)
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ 1)
  (h₃ : 0 ≤ x₃ ∧ x₃ ≤ 1)
  (h₄ : 0 ≤ x₄ ∧ x₄ ≤ 1) :
  (∀ y₁ y₂ y₃ y₄ : ℝ,
    0 ≤ y₁ ∧ y₁ ≤ 1 →
    0 ≤ y₂ ∧ y₂ ≤ 1 →
    0 ≤ y₃ ∧ y₃ ≤ 1 →
    0 ≤ y₄ ∧ y₄ ≤ 1 →
    1 - (1 - y₁) * (1 - y₂) * (1 - y₃) * (1 - y₄) ≤ 1 - (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄)) ∧
  1 - (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2715_271514


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2715_271510

def number_of_books : ℕ := 6
def number_of_identical_pairs : ℕ := 2
def books_per_pair : ℕ := 2

theorem book_arrangement_count :
  (number_of_books.factorial) / ((books_per_pair.factorial) ^ number_of_identical_pairs) = 180 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2715_271510


namespace NUMINAMATH_CALUDE_expression_equals_36_l2715_271531

theorem expression_equals_36 (k : ℚ) : k = 13 → k * (3 - 3 / k) = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_36_l2715_271531


namespace NUMINAMATH_CALUDE_discounted_price_theorem_l2715_271554

/-- The actual price of the good before discounts -/
def actual_price : ℝ := 9356.725146198829

/-- The first discount rate -/
def discount1 : ℝ := 0.20

/-- The second discount rate -/
def discount2 : ℝ := 0.10

/-- The third discount rate -/
def discount3 : ℝ := 0.05

/-- The final selling price after all discounts -/
def final_price : ℝ := 6400

/-- Theorem stating that applying the successive discounts to the actual price results in the final price -/
theorem discounted_price_theorem :
  actual_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = final_price := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_theorem_l2715_271554


namespace NUMINAMATH_CALUDE_stratified_sample_second_year_l2715_271505

/-- Represents the number of students in each year of high school -/
structure HighSchool where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the total number of students in the high school -/
def total_students (hs : HighSchool) : ℕ :=
  hs.first_year + hs.second_year + hs.third_year

/-- Calculates the number of students from a specific year in a stratified sample -/
def stratified_sample_size (hs : HighSchool) (year_size : ℕ) (sample_size : ℕ) : ℕ :=
  (year_size * sample_size) / total_students hs

/-- Theorem: In a stratified sample of 100 students from a high school with 1000 first-year,
    800 second-year, and 700 third-year students, the number of second-year students
    in the sample is 32. -/
theorem stratified_sample_second_year :
  let hs : HighSchool := ⟨1000, 800, 700⟩
  stratified_sample_size hs hs.second_year 100 = 32 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_second_year_l2715_271505


namespace NUMINAMATH_CALUDE_product_of_equal_sums_l2715_271596

theorem product_of_equal_sums (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_equal_sums_l2715_271596


namespace NUMINAMATH_CALUDE_max_b_value_l2715_271500

/-- The volume of the box -/
def volume : ℕ := 360

/-- Theorem: Given a box with volume 360 cubic units and dimensions a, b, and c,
    where a, b, and c are integers satisfying 1 < c < b < a,
    the maximum possible value of b is 12. -/
theorem max_b_value (a b c : ℕ) 
  (h_volume : a * b * c = volume)
  (h_order : 1 < c ∧ c < b ∧ b < a) :
  b ≤ 12 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = volume ∧ 1 < c' ∧ c' < b' ∧ b' < a' ∧ b' = 12 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l2715_271500


namespace NUMINAMATH_CALUDE_kath_group_cost_l2715_271522

/-- Calculates the total cost for a group watching a movie with early showing discount -/
def total_cost (standard_price : ℕ) (discount : ℕ) (group_size : ℕ) : ℕ :=
  (standard_price - discount) * group_size

/-- Theorem: The total cost for Kath's group is $30 -/
theorem kath_group_cost :
  let standard_price : ℕ := 8
  let early_discount : ℕ := 3
  let group_size : ℕ := 6
  total_cost standard_price early_discount group_size = 30 := by
  sorry

end NUMINAMATH_CALUDE_kath_group_cost_l2715_271522


namespace NUMINAMATH_CALUDE_marts_income_percentage_l2715_271529

/-- Given that Tim's income is 60 percent less than Juan's income
    and Mart's income is 64 percent of Juan's income,
    prove that Mart's income is 60 percent more than Tim's income. -/
theorem marts_income_percentage (juan tim mart : ℝ) 
  (h1 : tim = juan - 0.60 * juan)
  (h2 : mart = 0.64 * juan) :
  mart = tim + 0.60 * tim := by
  sorry

end NUMINAMATH_CALUDE_marts_income_percentage_l2715_271529


namespace NUMINAMATH_CALUDE_a_power_sum_l2715_271520

theorem a_power_sum (a x : ℝ) (ha : a > 0) (hx : a^(x/2) + a^(-x/2) = 5) : 
  a^x + a^(-x) = 23 := by
sorry

end NUMINAMATH_CALUDE_a_power_sum_l2715_271520


namespace NUMINAMATH_CALUDE_train_crossing_time_l2715_271527

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_length : Real) (signal_cross_time : Real) (platform_length : Real) :
  train_length = 300 ∧ 
  signal_cross_time = 16 ∧ 
  platform_length = 431.25 →
  (train_length + platform_length) / (train_length / signal_cross_time) = 39 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2715_271527


namespace NUMINAMATH_CALUDE_closest_integer_to_k_l2715_271561

theorem closest_integer_to_k : ∃ (k : ℝ), 
  k = Real.sqrt 2 * ((Real.sqrt 5 + Real.sqrt 3) * (Real.sqrt 5 - Real.sqrt 3)) ∧
  ∀ (n : ℤ), |k - 3| ≤ |k - n| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_k_l2715_271561


namespace NUMINAMATH_CALUDE_circle_symmetry_l2715_271558

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

/-- Definition of the symmetry line -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- Definition of symmetry between points with respect to the line -/
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = y₁ + 1 ∧ y₂ = x₁ - 1

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

/-- Theorem stating that C₂ is symmetric to C₁ with respect to the given line -/
theorem circle_symmetry :
  ∀ x y : ℝ, C₂ x y ↔ ∃ x₁ y₁ : ℝ, C₁ x₁ y₁ ∧ symmetric_points x₁ y₁ x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2715_271558


namespace NUMINAMATH_CALUDE_lcm_12_18_l2715_271559

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l2715_271559


namespace NUMINAMATH_CALUDE_bathroom_visit_interval_l2715_271571

/-- Calculates the time between bathroom visits during a movie -/
theorem bathroom_visit_interval (movie_duration : Real) (visit_count : Nat) : 
  movie_duration = 2.5 ∧ visit_count = 3 → 
  (movie_duration * 60) / (visit_count + 1) = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_visit_interval_l2715_271571


namespace NUMINAMATH_CALUDE_box_max_volume_l2715_271536

/-- The volume of the box as a function of the side length of the cut squares -/
def boxVolume (x : ℝ) : ℝ := (10 - 2*x) * (16 - 2*x) * x

/-- The maximum volume of the box -/
def maxVolume : ℝ := 144

theorem box_max_volume :
  ∃ (x : ℝ), 0 < x ∧ x < 5 ∧ 
  (∀ (y : ℝ), 0 < y ∧ y < 5 → boxVolume y ≤ boxVolume x) ∧
  boxVolume x = maxVolume :=
sorry

end NUMINAMATH_CALUDE_box_max_volume_l2715_271536


namespace NUMINAMATH_CALUDE_tan_70_plus_tan_50_minus_sqrt3_tan_70_tan_50_l2715_271516

theorem tan_70_plus_tan_50_minus_sqrt3_tan_70_tan_50 :
  Real.tan (70 * π / 180) + Real.tan (50 * π / 180) - Real.sqrt 3 * Real.tan (70 * π / 180) * Real.tan (50 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_70_plus_tan_50_minus_sqrt3_tan_70_tan_50_l2715_271516


namespace NUMINAMATH_CALUDE_number_forms_and_products_l2715_271546

theorem number_forms_and_products (n m : ℕ) :
  -- Part 1: Any number not divisible by 2 or 3 is of the form 6n + 1 or 6n + 5
  (∀ k : ℤ, (¬(2 ∣ k) ∧ ¬(3 ∣ k)) → (∃ n : ℕ, k = 6*n + 1 ∨ k = 6*n + 5)) ∧
  
  -- Part 2: Product of two numbers of form 6n + 1 or 6n + 5 is of form 6m + 1
  ((6*n + 1) * (6*m + 1) ≡ 1 [MOD 6]) ∧
  ((6*n + 5) * (6*m + 5) ≡ 1 [MOD 6]) ∧
  
  -- Part 3: Product of 6n + 1 and 6n + 5 is of form 6m + 5
  ((6*n + 1) * (6*m + 5) ≡ 5 [MOD 6]) :=
by sorry


end NUMINAMATH_CALUDE_number_forms_and_products_l2715_271546


namespace NUMINAMATH_CALUDE_spring_mass_for_27cm_unique_mass_for_27cm_l2715_271577

-- Define the relationship between spring length and mass
def spring_length (mass : ℝ) : ℝ := 16 + 2 * mass

-- Theorem stating that when the spring length is 27 cm, the mass is 5.5 kg
theorem spring_mass_for_27cm : spring_length 5.5 = 27 := by
  sorry

-- Theorem stating the uniqueness of the solution
theorem unique_mass_for_27cm (mass : ℝ) : 
  spring_length mass = 27 → mass = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_spring_mass_for_27cm_unique_mass_for_27cm_l2715_271577


namespace NUMINAMATH_CALUDE_percent_relation_l2715_271512

theorem percent_relation (x y z : ℝ) 
  (h1 : x = y * 1.2)  -- x is 20 percent more than y
  (h2 : y = z * 0.7)  -- y is 30 percent less than z
  : x = z * 0.84 :=   -- x is 84 percent of z
by sorry

end NUMINAMATH_CALUDE_percent_relation_l2715_271512


namespace NUMINAMATH_CALUDE_line_vector_at_5_l2715_271551

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_at_5 :
  (∀ t : ℝ, ∃ x y z : ℝ, line_vector t = (x, y, z)) →
  line_vector (-1) = (2, 6, 16) →
  line_vector 1 = (1, 3, 8) →
  line_vector 4 = (-2, -6, -16) →
  line_vector 5 = (-4, -12, -8) := by
  sorry

end NUMINAMATH_CALUDE_line_vector_at_5_l2715_271551


namespace NUMINAMATH_CALUDE_bus_time_is_ten_l2715_271526

/-- Represents the travel times and conditions for Xiaoming's journey --/
structure TravelTimes where
  total : ℕ  -- Total travel time
  transfer : ℕ  -- Transfer time
  subway_only : ℕ  -- Time if only taking subway
  bus_only : ℕ  -- Time if only taking bus

/-- Calculates the time spent on the bus given the travel times --/
def time_on_bus (t : TravelTimes) : ℕ :=
  let actual_travel_time := t.total - t.transfer
  let extra_time := actual_travel_time - t.subway_only
  let time_unit := extra_time / (t.bus_only / 10 - t.subway_only / 10)
  (t.bus_only / 10) * time_unit

/-- Theorem stating that given the specific travel times, the time spent on the bus is 10 minutes --/
theorem bus_time_is_ten : 
  let t : TravelTimes := { 
    total := 40, 
    transfer := 6, 
    subway_only := 30, 
    bus_only := 50 
  }
  time_on_bus t = 10 := by
  sorry


end NUMINAMATH_CALUDE_bus_time_is_ten_l2715_271526


namespace NUMINAMATH_CALUDE_fourth_root_of_four_sixes_l2715_271502

theorem fourth_root_of_four_sixes : 
  (4^6 + 4^6 + 4^6 + 4^6 : ℝ)^(1/4) = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_four_sixes_l2715_271502


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l2715_271539

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ
  black : ℕ
  yellow : ℕ
  blue : ℕ

/-- The probability of drawing a yellow marble as the second marble -/
def second_yellow_probability (bagA bagB bagC : Bag) : ℚ :=
  let total_A := bagA.white + bagA.black
  let total_B := bagB.yellow + bagB.blue
  let total_C := bagC.yellow + bagC.blue
  let prob_white_A : ℚ := bagA.white / total_A
  let prob_black_A : ℚ := bagA.black / total_A
  let prob_yellow_B : ℚ := bagB.yellow / total_B
  let prob_yellow_C : ℚ := bagC.yellow / total_C
  prob_white_A * prob_yellow_B + prob_black_A * prob_yellow_C

/-- The main theorem stating the probability of drawing a yellow marble as the second marble -/
theorem yellow_marble_probability :
  let bagA : Bag := ⟨3, 4, 0, 0⟩
  let bagB : Bag := ⟨0, 0, 6, 4⟩
  let bagC : Bag := ⟨0, 0, 2, 5⟩
  second_yellow_probability bagA bagB bagC = 103 / 245 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marble_probability_l2715_271539


namespace NUMINAMATH_CALUDE_positive_distinct_solutions_l2715_271579

/-- Given a system of equations, prove the necessary and sufficient conditions for positive and distinct solutions -/
theorem positive_distinct_solutions (a b x y z : ℝ) :
  x + y + z = a →
  x^2 + y^2 + z^2 = b^2 →
  x * y = z^2 →
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ (a > 0 ∧ b^2 < a^2 ∧ a^2 < 3 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_positive_distinct_solutions_l2715_271579


namespace NUMINAMATH_CALUDE_circle_circumference_after_scaling_l2715_271568

theorem circle_circumference_after_scaling (a b : ℝ) (h1 : a = 7) (h2 : b = 24) : 
  let d := Real.sqrt (a^2 + b^2)
  let new_d := 1.5 * d
  let new_circumference := π * new_d
  new_circumference = 37.5 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_after_scaling_l2715_271568


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l2715_271562

/-- Given the cost of 3 pens and 5 pencils, and the cost ratio of pen to pencil,
    prove the cost of one dozen pens -/
theorem cost_of_dozen_pens (cost_3pens_5pencils : ℕ) (ratio_pen_pencil : ℚ) :
  cost_3pens_5pencils = 240 →
  ratio_pen_pencil = 5 / 1 →
  (12 : ℕ) * (5 * (cost_3pens_5pencils / (3 * 5 + 5))) = 720 := by
sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l2715_271562


namespace NUMINAMATH_CALUDE_pigeonhole_divisibility_l2715_271594

theorem pigeonhole_divisibility (n : ℕ+) (a : Fin (n + 1) → ℤ) :
  ∃ i j : Fin (n + 1), i ≠ j ∧ (n : ℤ) ∣ (a i - a j) := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_divisibility_l2715_271594


namespace NUMINAMATH_CALUDE_cosine_in_special_triangle_l2715_271528

/-- 
In a triangle ABC, given that:
1. The side lengths a, b, c form a geometric sequence
2. c = 2a
Then, cos B = 3/4
-/
theorem cosine_in_special_triangle (a b c : ℝ) (h_positive : a > 0) 
  (h_geometric : b^2 = a * c) (h_relation : c = 2 * a) :
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  cos_B = 3/4 := by sorry

end NUMINAMATH_CALUDE_cosine_in_special_triangle_l2715_271528


namespace NUMINAMATH_CALUDE_marin_apples_l2715_271578

theorem marin_apples (donald_apples : ℕ) (total_apples : ℕ) 
  (h1 : donald_apples = 2)
  (h2 : total_apples = 11) :
  ∃ marin_apples : ℕ, marin_apples + donald_apples = total_apples ∧ marin_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_marin_apples_l2715_271578


namespace NUMINAMATH_CALUDE_min_value_of_s_l2715_271573

theorem min_value_of_s (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 3 * x^2 + 2 * y^2 + z^2 = 1) :
  (1 + z) / (x * y * z) ≥ 8 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_s_l2715_271573


namespace NUMINAMATH_CALUDE_partitioned_square_theorem_main_theorem_l2715_271598

/-- A square with interior points and partitioned into triangles -/
structure PartitionedSquare where
  /-- The number of interior points in the square -/
  num_interior_points : ℕ
  /-- The number of line segments drawn -/
  num_segments : ℕ
  /-- The number of triangles formed -/
  num_triangles : ℕ
  /-- Ensures that the number of interior points is 1965 -/
  h_points : num_interior_points = 1965

/-- Theorem stating the relationship between the number of interior points,
    line segments, and triangles in a partitioned square -/
theorem partitioned_square_theorem (ps : PartitionedSquare) :
  ps.num_segments = 5896 ∧ ps.num_triangles = 3932 := by
  sorry

/-- Main theorem proving the specific case for 1965 interior points -/
theorem main_theorem : 
  ∃ ps : PartitionedSquare, ps.num_segments = 5896 ∧ ps.num_triangles = 3932 := by
  sorry

end NUMINAMATH_CALUDE_partitioned_square_theorem_main_theorem_l2715_271598


namespace NUMINAMATH_CALUDE_complex_number_imaginary_part_l2715_271521

theorem complex_number_imaginary_part (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  Complex.im z = 1 → a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_number_imaginary_part_l2715_271521


namespace NUMINAMATH_CALUDE_nine_sided_polygon_odd_spanning_diagonals_l2715_271515

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- We don't need to define the specifics of a regular polygon for this problem

/-- The number of diagonals in a regular polygon that span an odd number of vertices between their endpoints -/
def oddSpanningDiagonals (p : RegularPolygon n) : ℕ :=
  sorry  -- Definition to be implemented

/-- Theorem stating that a regular nine-sided polygon has 18 diagonals spanning an odd number of vertices -/
theorem nine_sided_polygon_odd_spanning_diagonals :
  ∀ (p : RegularPolygon 9), oddSpanningDiagonals p = 18 :=
by sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_odd_spanning_diagonals_l2715_271515


namespace NUMINAMATH_CALUDE_circle_properties_l2715_271552

/-- Represents a circle in the 2D plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Calculates the center of a circle given its equation -/
def circle_center (c : Circle) : ℝ × ℝ := sorry

/-- Calculates the length of the shortest chord passing through a given point -/
def shortest_chord_length (c : Circle) (p : ℝ × ℝ) : ℝ := sorry

/-- The main theorem about the circle and its properties -/
theorem circle_properties :
  let c : Circle := { equation := fun x y => x^2 + y^2 - 6*x - 8*y = 0 }
  let p : ℝ × ℝ := (3, 5)
  circle_center c = (3, 4) ∧
  shortest_chord_length c p = 4 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_circle_properties_l2715_271552


namespace NUMINAMATH_CALUDE_emily_bought_two_skirts_l2715_271511

def cost_of_art_supplies : ℕ := 20
def total_spent : ℕ := 50
def cost_per_skirt : ℕ := 15

def number_of_skirts : ℕ := (total_spent - cost_of_art_supplies) / cost_per_skirt

theorem emily_bought_two_skirts : number_of_skirts = 2 := by
  sorry

end NUMINAMATH_CALUDE_emily_bought_two_skirts_l2715_271511


namespace NUMINAMATH_CALUDE_unique_solution_l2715_271506

/-- Given denominations 3, n, and n+2, returns true if m cents can be formed -/
def can_form_postage (n : ℕ) (m : ℕ) : Prop :=
  ∃ (a b c : ℕ), m = 3 * a + n * b + (n + 2) * c

/-- Returns true if n satisfies the problem conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  (∀ m > 63, can_form_postage n m) ∧
  ¬(can_form_postage n 63)

theorem unique_solution :
  ∃! n : ℕ, n > 0 ∧ satisfies_conditions n ∧ n = 30 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2715_271506


namespace NUMINAMATH_CALUDE_set_A_proof_l2715_271545

def U : Set ℕ := {1, 3, 5, 7, 9}

theorem set_A_proof (A B : Set ℕ) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : A ∩ B = {3, 5})
  (h4 : A ∩ (U \ B) = {9}) :
  A = {3, 5, 9} := by
  sorry


end NUMINAMATH_CALUDE_set_A_proof_l2715_271545


namespace NUMINAMATH_CALUDE_martha_children_count_l2715_271535

theorem martha_children_count (total_cakes : ℕ) (cakes_per_child : ℕ) (h1 : total_cakes = 18) (h2 : cakes_per_child = 6) : 
  total_cakes / cakes_per_child = 3 :=
by sorry

end NUMINAMATH_CALUDE_martha_children_count_l2715_271535


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l2715_271508

-- Define the slopes of the two lines
def slope1 : ℚ := 3 / 4
def slope2 (b : ℚ) : ℚ := -b / 2

-- Define the perpendicularity condition
def perpendicular (b : ℚ) : Prop := slope1 * slope2 b = -1

-- Theorem statement
theorem perpendicular_lines_b_value :
  ∃ b : ℚ, perpendicular b ∧ b = 8 / 3 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l2715_271508


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l2715_271587

theorem average_of_three_numbers (x : ℝ) : 
  (12 + 21 + x) / 3 = 18 → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l2715_271587


namespace NUMINAMATH_CALUDE_eggs_and_cakes_l2715_271589

def dozen : ℕ := 12

def initial_eggs : ℕ := 7 * dozen
def used_eggs : ℕ := 5 * dozen
def eggs_per_cake : ℕ := (3 * dozen) / 2

theorem eggs_and_cakes :
  let remaining_eggs := initial_eggs - used_eggs
  let possible_cakes := remaining_eggs / eggs_per_cake
  remaining_eggs = 24 ∧ possible_cakes = 1 := by sorry

end NUMINAMATH_CALUDE_eggs_and_cakes_l2715_271589


namespace NUMINAMATH_CALUDE_fourth_month_sale_l2715_271582

theorem fourth_month_sale 
  (sale1 sale2 sale3 sale5 sale6 average_sale : ℕ)
  (h1 : sale1 = 5435)
  (h2 : sale2 = 5927)
  (h3 : sale3 = 5855)
  (h5 : sale5 = 5562)
  (h6 : sale6 = 3991)
  (h_avg : average_sale = 5500)
  : ∃ sale4 : ℕ, sale4 = 6230 ∧ 
    (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale := by
  sorry

#check fourth_month_sale

end NUMINAMATH_CALUDE_fourth_month_sale_l2715_271582


namespace NUMINAMATH_CALUDE_scientific_notation_pm25_l2715_271503

theorem scientific_notation_pm25 :
  ∃ (a : ℝ) (n : ℤ), 0.000042 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.2 ∧ n = -5 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_pm25_l2715_271503


namespace NUMINAMATH_CALUDE_unique_three_digit_numbers_l2715_271517

theorem unique_three_digit_numbers : ∃! (x y : ℕ), 
  100 ≤ x ∧ x ≤ 999 ∧ 
  100 ≤ y ∧ y ≤ 999 ∧ 
  1000 * x + y = 7 * x * y ∧
  x = 143 ∧ y = 143 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_numbers_l2715_271517


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2715_271541

theorem unique_solution_quadratic (q : ℝ) : 
  (q ≠ 0 ∧ ∀ x : ℝ, (q * x^2 - 18 * x + 8 = 0 → (∀ y : ℝ, q * y^2 - 18 * y + 8 = 0 → x = y))) ↔ 
  q = 81/8 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2715_271541


namespace NUMINAMATH_CALUDE_factorization_proof_l2715_271523

theorem factorization_proof (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2715_271523


namespace NUMINAMATH_CALUDE_external_diagonals_invalid_l2715_271563

theorem external_diagonals_invalid (a b c : ℝ) : 
  a = 4 ∧ b = 6 ∧ c = 8 →
  ¬(a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ a^2 + c^2 > b^2) :=
by sorry

end NUMINAMATH_CALUDE_external_diagonals_invalid_l2715_271563


namespace NUMINAMATH_CALUDE_log_inequality_relation_l2715_271519

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_inequality_relation :
  (∀ x y : ℝ, x > 0 → y > 0 → (log x < log y → x < y)) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x < y ∧ ¬(log x < log y)) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_relation_l2715_271519


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2715_271524

theorem quadratic_roots_sum (α β : ℝ) : 
  α^2 + 2*α - 2024 = 0 → 
  β^2 + 2*β - 2024 = 0 → 
  α^2 + 3*α + β = 2022 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2715_271524


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_eq_three_fourths_l2715_271557

/-- The equation (x - 3) / (ax - 2) = x has exactly one solution if and only if a = 3/4 -/
theorem unique_solution_iff_a_eq_three_fourths (a : ℝ) : 
  (∃! x : ℝ, (x - 3) / (a * x - 2) = x) ↔ a = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_eq_three_fourths_l2715_271557


namespace NUMINAMATH_CALUDE_root_implies_difference_of_fourth_powers_l2715_271543

theorem root_implies_difference_of_fourth_powers (a b : ℝ) :
  (∃ x, x^2 - 4*a^2*b^2*x = 4 ∧ x = (a^2 + b^2)^2) →
  (a^4 - b^4 = 2 ∨ a^4 - b^4 = -2) :=
by sorry

end NUMINAMATH_CALUDE_root_implies_difference_of_fourth_powers_l2715_271543


namespace NUMINAMATH_CALUDE_point_transformation_l2715_271560

def rotate90CCW (x y : ℝ) : ℝ × ℝ := (-y, x)

def reflectAboutYeqX (x y : ℝ) : ℝ × ℝ := (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate90CCW a b
  let (x₂, y₂) := reflectAboutYeqX x₁ y₁
  (x₂ = 3 ∧ y₂ = -7) → b - a = 4 := by sorry

end NUMINAMATH_CALUDE_point_transformation_l2715_271560
