import Mathlib

namespace triangle_sphere_distance_l2545_254542

/-- The distance between the plane containing a triangle inscribed on a sphere and the center of the sphere -/
theorem triangle_sphere_distance (a b c R : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) (hR : R = 10) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := area / s
  Real.sqrt (R^2 - r^2) = 2 * Real.sqrt 21 :=
by sorry

end triangle_sphere_distance_l2545_254542


namespace wedding_rsvp_yes_percentage_l2545_254544

theorem wedding_rsvp_yes_percentage 
  (total_guests : ℕ) 
  (no_response_percentage : ℚ) 
  (no_reply_guests : ℕ) : 
  total_guests = 200 →
  no_response_percentage = 9 / 100 →
  no_reply_guests = 16 →
  (↑(total_guests - (total_guests * no_response_percentage).floor - no_reply_guests) / total_guests : ℚ) = 83 / 100 := by
sorry

end wedding_rsvp_yes_percentage_l2545_254544


namespace common_tangent_sum_l2545_254551

/-- A line y = kx + b is a common tangent to the curves y = ln(1+x) and y = 2 + ln(x) -/
def isCommonTangent (k b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, 
    (k * x₁ + b = Real.log (1 + x₁)) ∧
    (k * x₂ + b = 2 + Real.log x₂) ∧
    (k = 1 / (1 + x₁)) ∧
    (k = 1 / x₂)

/-- If a line y = kx + b is a common tangent to the curves y = ln(1+x) and y = 2 + ln(x), 
    then k + b = 3 - ln(2) -/
theorem common_tangent_sum (k b : ℝ) : 
  isCommonTangent k b → k + b = 3 - Real.log 2 := by
  sorry

end common_tangent_sum_l2545_254551


namespace intersection_singleton_complement_intersection_l2545_254589

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}

-- Part 1
theorem intersection_singleton (a : ℝ) : A ∩ B a = {2} → a = -1 ∨ a = -3 := by
  sorry

-- Part 2
theorem complement_intersection (a : ℝ) : A ∩ (Set.univ \ B a) = A →
  a < -3 ∨ (-3 < a ∧ a < -1 - Real.sqrt 3) ∨
  (-1 - Real.sqrt 3 < a ∧ a < -1) ∨
  (-1 < a ∧ a < -1 + Real.sqrt 3) ∨
  a > -1 + Real.sqrt 3 := by
  sorry

end intersection_singleton_complement_intersection_l2545_254589


namespace range_a_theorem_l2545_254514

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (a > -2 ∧ a < -1) ∨ a ≥ 1

-- Theorem statement
theorem range_a_theorem (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_of_a a :=
sorry

end range_a_theorem_l2545_254514


namespace range_of_m_l2545_254525

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → x^2 - 4*x - 2*m + 1 ≤ 0) → m ≥ 3 := by
  sorry

end range_of_m_l2545_254525


namespace megan_popsicle_consumption_l2545_254550

/-- The number of Popsicles Megan can finish in a given time -/
def popsicles_finished (popsicle_interval : ℕ) (total_time : ℕ) : ℕ :=
  total_time / popsicle_interval

theorem megan_popsicle_consumption :
  let popsicle_interval : ℕ := 15  -- minutes
  let hours : ℕ := 4
  let additional_minutes : ℕ := 30
  let total_time : ℕ := hours * 60 + additional_minutes
  popsicles_finished popsicle_interval total_time = 18 := by
  sorry

end megan_popsicle_consumption_l2545_254550


namespace line_passes_through_point_l2545_254578

/-- A line passing through a point -/
def line_passes_through (k : ℝ) (x y : ℝ) : Prop :=
  2 - k * x = -5 * y

/-- The theorem stating that the line passes through the given point when k = -0.5 -/
theorem line_passes_through_point :
  line_passes_through (-0.5) 6 (-1) := by sorry

end line_passes_through_point_l2545_254578


namespace max_value_expression_l2545_254593

theorem max_value_expression (a b c : ℝ) 
  (ha : -1 ≤ a ∧ a ≤ 1) 
  (hb : -1 ≤ b ∧ b ≤ 1) 
  (hc : -1 ≤ c ∧ c ≤ 1) : 
  ∀ x y z : ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ y ∧ y ≤ 1 → -1 ≤ z ∧ z ≤ 1 → 
  2 * Real.sqrt (a * b * c) + Real.sqrt ((1 - a^2) * (1 - b^2) * (1 - c^2)) ≤ 
  2 * Real.sqrt (x * y * z) + Real.sqrt ((1 - x^2) * (1 - y^2) * (1 - z^2)) → 
  2 * Real.sqrt (x * y * z) + Real.sqrt ((1 - x^2) * (1 - y^2) * (1 - z^2)) ≤ 2 :=
by sorry

end max_value_expression_l2545_254593


namespace smallest_k_inequality_l2545_254559

theorem smallest_k_inequality (k : ℝ) : k = 1 ↔ 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + k * (x - y)^2 ≥ Real.sqrt (x^2 + y^2)) ∧ 
  (∀ k' : ℝ, k' > 0 → k' < k → ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ Real.sqrt (x * y) + k' * (x - y)^2 < Real.sqrt (x^2 + y^2)) :=
by sorry

end smallest_k_inequality_l2545_254559


namespace mary_regular_hours_l2545_254574

/-- Mary's work schedule and pay structure --/
structure MaryWork where
  max_hours : ℕ
  regular_rate : ℚ
  overtime_rate : ℚ
  total_earnings : ℚ

/-- Theorem stating Mary's regular work hours --/
theorem mary_regular_hours (w : MaryWork) 
  (h1 : w.max_hours = 60)
  (h2 : w.regular_rate = 8)
  (h3 : w.overtime_rate = w.regular_rate * (1 + 1/4))
  (h4 : w.total_earnings = 560) :
  ∃ (regular_hours overtime_hours : ℕ),
    regular_hours + overtime_hours = w.max_hours ∧
    regular_hours * w.regular_rate + overtime_hours * w.overtime_rate = w.total_earnings ∧
    regular_hours = 20 := by
  sorry

end mary_regular_hours_l2545_254574


namespace sandy_sums_attempted_sandy_specific_case_l2545_254530

theorem sandy_sums_attempted (marks_per_correct : ℕ) (marks_per_incorrect : ℕ) 
  (total_marks : ℕ) (correct_sums : ℕ) : ℕ :=
  let total_sums := correct_sums + (marks_per_correct * correct_sums - total_marks) / marks_per_incorrect
  total_sums

theorem sandy_specific_case : sandy_sums_attempted 3 2 65 25 = 30 := by
  sorry

end sandy_sums_attempted_sandy_specific_case_l2545_254530


namespace max_edges_100_vertices_triangle_free_l2545_254520

/-- The maximum number of edges in a triangle-free graph with n vertices -/
def maxEdgesTriangleFree (n : ℕ) : ℕ := n^2 / 4

/-- Theorem: In a graph with 100 vertices and no triangles, the maximum number of edges is 2500 -/
theorem max_edges_100_vertices_triangle_free :
  maxEdgesTriangleFree 100 = 2500 := by
  sorry

#eval maxEdgesTriangleFree 100  -- Should output 2500

end max_edges_100_vertices_triangle_free_l2545_254520


namespace parabola_intersection_properties_l2545_254540

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2
def g (x : ℝ) : ℝ := 2 * x - 3
def h (x : ℝ) : ℝ := 2

-- Define the theorem
theorem parabola_intersection_properties
  (a : ℝ)
  (ha : a ≠ 0)
  (h_intersection : f a 1 = g 1) :
  (a = -1) ∧
  (∀ x, f a x = -x^2) ∧
  (∀ x, x < 0 → (∀ y, y < x → f a y < f a x)) ∧
  (let x₁ := Real.sqrt 2
   let x₂ := -Real.sqrt 2
   let area := (1/2) * (x₁ - x₂) * (h x₁ - f a 0)
   area = 2 * Real.sqrt 2) :=
by sorry

end parabola_intersection_properties_l2545_254540


namespace intersection_of_M_and_N_l2545_254563

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end intersection_of_M_and_N_l2545_254563


namespace infinitely_many_divisible_by_m_l2545_254586

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- State the theorem
theorem infinitely_many_divisible_by_m (m : ℤ) :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ m ∣ fib n :=
sorry

end infinitely_many_divisible_by_m_l2545_254586


namespace office_average_age_l2545_254557

/-- The average age of all persons in an office, given specific conditions -/
theorem office_average_age :
  let total_persons : ℕ := 18
  let group1_size : ℕ := 5
  let group1_avg : ℚ := 14
  let group2_size : ℕ := 9
  let group2_avg : ℚ := 16
  let person15_age : ℕ := 56
  (total_persons : ℚ) * (average_age : ℚ) =
    (group1_size : ℚ) * group1_avg +
    (group2_size : ℚ) * group2_avg +
    (person15_age : ℚ) +
    ((total_persons - group1_size - group2_size - 1) : ℚ) * average_age →
  average_age = 270 / 14 := by
sorry

end office_average_age_l2545_254557


namespace rectangle_diagonals_not_always_perpendicular_and_equal_l2545_254504

/-- A rectangle is a quadrilateral with four right angles -/
structure Rectangle where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  is_right_angle : ∀ i, angles i = π / 2

/-- Diagonals of a shape -/
def diagonals (r : Rectangle) : Fin 2 → ℝ := sorry

/-- Two real numbers are equal -/
def are_equal (a b : ℝ) : Prop := a = b

/-- Two lines are perpendicular if they form a right angle -/
def are_perpendicular (a b : ℝ) : Prop := sorry

theorem rectangle_diagonals_not_always_perpendicular_and_equal (r : Rectangle) : 
  ¬(are_equal (diagonals r 0) (diagonals r 1) ∧ are_perpendicular (diagonals r 0) (diagonals r 1)) := by
  sorry

end rectangle_diagonals_not_always_perpendicular_and_equal_l2545_254504


namespace product_of_numbers_l2545_254581

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x - y = 4) (h3 : x / y = 5 / 4) :
  x * y = 320 := by
  sorry

end product_of_numbers_l2545_254581


namespace group_size_calculation_l2545_254536

theorem group_size_calculation (iceland : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : iceland = 55)
  (h2 : norway = 43)
  (h3 : both = 61)
  (h4 : neither = 63) :
  iceland + norway - both + neither = 161 :=
by sorry

end group_size_calculation_l2545_254536


namespace coltons_marbles_l2545_254553

theorem coltons_marbles (white_marbles : ℕ) : 
  (∃ (groups : ℕ), groups = 8 ∧ (16 + white_marbles) % groups = 0) →
  ∃ (k : ℕ), white_marbles = 8 * k :=
by sorry

end coltons_marbles_l2545_254553


namespace trigonometric_identities_l2545_254529

theorem trigonometric_identities (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.sin α = 3 / 5) : 
  ((2 * Real.sin α ^ 2 + Real.sin (2 * α)) / Real.cos (2 * α) = 24 / 7) ∧ 
  (Real.tan (α + 5 * Real.pi / 4) = 7) := by
  sorry

end trigonometric_identities_l2545_254529


namespace twenty_eighth_term_of_sequence_l2545_254582

def sequence_term (n : ℕ) : ℚ :=
  1 / (2 ^ (sum_of_repeated_terms n))
where
  sum_of_repeated_terms : ℕ → ℕ
  | 0 => 0
  | k + 1 => if (sum_of_repeated_terms k + k + 1 < n) then k + 1 else k

theorem twenty_eighth_term_of_sequence :
  sequence_term 28 = 1 / (2 ^ 7) :=
sorry

end twenty_eighth_term_of_sequence_l2545_254582


namespace sqrt_three_subset_M_l2545_254580

def M : Set ℝ := {x | x ≤ 3}

theorem sqrt_three_subset_M : {Real.sqrt 3} ⊆ M := by sorry

end sqrt_three_subset_M_l2545_254580


namespace m_divided_by_8_l2545_254522

theorem m_divided_by_8 (m : ℕ) (h : m = 16^1500) : m / 8 = 2^5997 := by
  sorry

end m_divided_by_8_l2545_254522


namespace simplify_fraction_expression_l2545_254508

theorem simplify_fraction_expression :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end simplify_fraction_expression_l2545_254508


namespace octal_addition_example_l2545_254534

/-- Represents a digit in the octal number system -/
def OctalDigit : Type := Fin 8

/-- Represents an octal number as a list of octal digits -/
def OctalNumber : Type := List OctalDigit

/-- Addition operation for octal numbers -/
def octal_add : OctalNumber → OctalNumber → OctalNumber :=
  sorry

/-- Conversion from a natural number to an octal number -/
def nat_to_octal : Nat → OctalNumber :=
  sorry

/-- Theorem: 47 + 56 = 125 in the octal number system -/
theorem octal_addition_example :
  octal_add (nat_to_octal 47) (nat_to_octal 56) = nat_to_octal 125 := by
  sorry

end octal_addition_example_l2545_254534


namespace train_length_l2545_254565

/-- Given a train with speed 72 km/hr crossing a 260 m platform in 26 seconds, prove its length is 260 m -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * 1000 / 3600 →
  platform_length = 260 →
  crossing_time = 26 →
  (train_speed * crossing_time) - platform_length = 260 := by
  sorry

end train_length_l2545_254565


namespace integer_expression_l2545_254566

/-- Binomial coefficient -/
def binomial (m l : ℕ) : ℕ := Nat.choose m l

/-- The main theorem -/
theorem integer_expression (l m : ℤ) (h1 : 1 ≤ l) (h2 : l < m) :
  ∃ (k : ℤ), ((m - 3*l + 2) / (l + 2)) * binomial m.toNat l.toNat = k ↔ 
  ∃ (n : ℤ), m + 8 = n * (l + 2) := by sorry

end integer_expression_l2545_254566


namespace increasing_function_parameter_range_l2545_254564

/-- Given a function f(x) = -1/3 * x^3 + 1/2 * x^2 + 2ax, 
    if f(x) is increasing on the interval (2/3, +∞), 
    then a ∈ (-1/9, +∞) -/
theorem increasing_function_parameter_range (a : ℝ) : 
  (∀ x > 2/3, (deriv (fun x => -1/3 * x^3 + 1/2 * x^2 + 2*a*x) x) > 0) →
  a > -1/9 :=
by sorry

end increasing_function_parameter_range_l2545_254564


namespace swimming_championship_races_swimming_championship_proof_l2545_254513

/-- Calculate the number of races needed to determine a champion in a swimming competition. -/
theorem swimming_championship_races (total_swimmers : ℕ) 
  (swimmers_per_race : ℕ) (advancing_swimmers : ℕ) : ℕ :=
  let eliminated_per_race := swimmers_per_race - advancing_swimmers
  let total_eliminations := total_swimmers - 1
  ⌈(total_eliminations : ℚ) / eliminated_per_race⌉.toNat

/-- Prove that 53 races are required for 300 swimmers with 8 per race and 2 advancing. -/
theorem swimming_championship_proof : 
  swimming_championship_races 300 8 2 = 53 := by
  sorry

end swimming_championship_races_swimming_championship_proof_l2545_254513


namespace arithmetic_sequence_problem_l2545_254552

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_correct : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h1 : seq.S 6 = 8 * seq.S 3)
    (h2 : seq.a 3 - seq.a 5 = 8) :
  seq.a 20 = -74 := by
  sorry

end arithmetic_sequence_problem_l2545_254552


namespace mets_to_red_sox_ratio_l2545_254519

/-- Proves that the ratio of NY Mets fans to Boston Red Sox fans is 4:5 given the conditions -/
theorem mets_to_red_sox_ratio :
  ∀ (yankees mets red_sox : ℕ),
  yankees + mets + red_sox = 360 →
  3 * mets = 2 * yankees →
  mets = 96 →
  5 * mets = 4 * red_sox :=
by
  sorry

end mets_to_red_sox_ratio_l2545_254519


namespace first_day_hike_distance_l2545_254577

/-- A hike with two participants -/
structure Hike where
  total_distance : ℕ
  distance_left : ℕ
  tripp_backpack_weight : ℕ
  charlotte_backpack_weight : ℕ
  (charlotte_lighter : charlotte_backpack_weight = tripp_backpack_weight - 7)

/-- The distance hiked on the first day -/
def distance_hiked_first_day (h : Hike) : ℕ :=
  h.total_distance - h.distance_left

theorem first_day_hike_distance (h : Hike) 
  (h_total : h.total_distance = 36) 
  (h_left : h.distance_left = 27) : 
  distance_hiked_first_day h = 9 := by
  sorry

end first_day_hike_distance_l2545_254577


namespace f_geq_one_solution_set_g_max_value_l2545_254594

def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

def g (x : ℝ) : ℝ := f x - x^2 + x

theorem f_geq_one_solution_set :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

theorem g_max_value :
  ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 5/4 := by sorry

end f_geq_one_solution_set_g_max_value_l2545_254594


namespace swamp_flies_eaten_l2545_254528

/-- Represents the number of animals in the swamp ecosystem -/
structure SwampPopulation where
  gharials : ℕ
  herons : ℕ
  caimans : ℕ
  fish : ℕ
  frogs : ℕ

/-- Calculates the total number of flies eaten daily in the swamp -/
def flies_eaten_daily (pop : SwampPopulation) : ℕ :=
  pop.frogs * 30 + pop.herons * 60

/-- Theorem stating the number of flies eaten daily in the given swamp ecosystem -/
theorem swamp_flies_eaten 
  (pop : SwampPopulation)
  (h_gharials : pop.gharials = 9)
  (h_herons : pop.herons = 12)
  (h_caimans : pop.caimans = 7)
  (h_fish : pop.fish = 20)
  (h_frogs : pop.frogs = 50) :
  flies_eaten_daily pop = 2220 := by
  sorry


end swamp_flies_eaten_l2545_254528


namespace triangle_properties_l2545_254517

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (Real.cos t.A) / (1 + Real.sin t.A) = (Real.sin (2 * t.B)) / (1 + Real.cos (2 * t.B)))
  (h2 : t.C = 2 * Real.pi / 3)
  (h3 : t.A + t.B + t.C = Real.pi)
  (h4 : t.a / Real.sin t.A = t.b / Real.sin t.B)
  (h5 : t.b / Real.sin t.B = t.c / Real.sin t.C)
  : 
  (t.B = Real.pi / 6) ∧ 
  (∀ (x : Triangle), x.A + x.B + x.C = Real.pi → 
    (x.a^2 + x.b^2) / x.c^2 ≥ 4 * Real.sqrt 2 - 5) :=
by sorry

end triangle_properties_l2545_254517


namespace cubic_polynomial_uniqueness_l2545_254568

/-- Given a cubic polynomial q(x) with the following properties:
    1) It has roots at 2, -2, and 1
    2) The function f(x) = (x^3 - 2x^2 - 5x + 6) / q(x) has no horizontal asymptote
    3) q(4) = 24
    Then q(x) = (2/3)x^3 - (2/3)x^2 - (8/3)x + 8/3 -/
theorem cubic_polynomial_uniqueness (q : ℝ → ℝ) :
  (∀ x, q x = 0 ↔ x = 2 ∨ x = -2 ∨ x = 1) →
  (∃ k, ∀ x, q x = k * (x - 2) * (x + 2) * (x - 1)) →
  q 4 = 24 →
  ∀ x, q x = (2/3) * x^3 - (2/3) * x^2 - (8/3) * x + 8/3 :=
by sorry

end cubic_polynomial_uniqueness_l2545_254568


namespace inequality_proof_l2545_254555

theorem inequality_proof (x : ℝ) : 4 ≤ (x + 1) / (3 * x - 7) ∧ (x + 1) / (3 * x - 7) < 9 ↔ x ∈ Set.Ioo (32 / 13) (29 / 11) := by
  sorry

end inequality_proof_l2545_254555


namespace triangle_inequality_variant_l2545_254521

theorem triangle_inequality_variant (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end triangle_inequality_variant_l2545_254521


namespace pencil_distribution_ways_l2545_254560

/-- The number of ways to distribute n identical objects into k distinct groups --/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute pencils among friends --/
def distributePencils (totalPencils friendCount minPencils : ℕ) : ℕ :=
  let remainingPencils := totalPencils - friendCount * minPencils
  starsAndBars remainingPencils friendCount

theorem pencil_distribution_ways :
  distributePencils 12 4 2 = 35 := by
  sorry

#eval distributePencils 12 4 2

end pencil_distribution_ways_l2545_254560


namespace safe_cracking_l2545_254501

def Password := Fin 10 → Fin 10

def isValidPassword (p : Password) : Prop :=
  (∀ i j : Fin 7, i ≠ j → p i ≠ p j) ∧ (∀ i : Fin 7, p i < 10)

def Attempt := Fin 7 → Fin 10

def isSuccessfulAttempt (p : Password) (a : Attempt) : Prop :=
  ∃ i : Fin 7, p i = a i

theorem safe_cracking (p : Password) (h : isValidPassword p) :
  ∃ attempts : Fin 6 → Attempt,
    ∃ i : Fin 6, isSuccessfulAttempt p (attempts i) :=
sorry

end safe_cracking_l2545_254501


namespace exist_good_numbers_counterexample_l2545_254549

/-- A natural number is "good" if its decimal representation contains only 0s and 1s -/
def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- Sum of digits of a natural number in base 10 -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Statement: There exist two good numbers whose product is good, but the sum of digits
    of their product is not equal to the product of their sums of digits -/
theorem exist_good_numbers_counterexample :
  ∃ (A B : ℕ), is_good A ∧ is_good B ∧ is_good (A * B) ∧
    sum_of_digits (A * B) ≠ sum_of_digits A * sum_of_digits B :=
sorry

end exist_good_numbers_counterexample_l2545_254549


namespace probability_red_black_white_l2545_254585

def total_balls : ℕ := 12
def red_balls : ℕ := 5
def black_balls : ℕ := 4
def white_balls : ℕ := 2
def green_balls : ℕ := 1

theorem probability_red_black_white :
  (red_balls + black_balls + white_balls : ℚ) / total_balls = 11 / 12 := by
  sorry

end probability_red_black_white_l2545_254585


namespace odd_function_value_l2545_254539

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the theorem
theorem odd_function_value (f g : ℝ → ℝ) 
  (h_odd : OddFunction f)
  (h_g : ∀ x, g x = f x + 6)
  (h_g_neg_one : g (-1) = 3) :
  f 1 = 3 := by
  sorry

end odd_function_value_l2545_254539


namespace min_sum_with_reciprocal_constraint_l2545_254506

theorem min_sum_with_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 9/y = 1) : 
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 9/y₀ = 1 ∧ x₀ + y₀ = 16 :=
sorry

end min_sum_with_reciprocal_constraint_l2545_254506


namespace line_through_points_sum_of_coefficients_l2545_254588

-- Define the line equation
def line_equation (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem line_through_points_sum_of_coefficients :
  ∀ a b : ℝ,
  line_equation a b 2 = 3 →
  line_equation a b 10 = 19 →
  a + b = 1 := by
  sorry


end line_through_points_sum_of_coefficients_l2545_254588


namespace p_or_q_and_not_p_implies_q_l2545_254576

theorem p_or_q_and_not_p_implies_q (p q : Prop) :
  (p ∨ q) → ¬p → q := by sorry

end p_or_q_and_not_p_implies_q_l2545_254576


namespace four_solutions_l2545_254570

/-- The number of integer pairs (m, n) satisfying (m-1)(n-1) = 2 -/
def count_solutions : ℕ := 4

/-- A pair of integers (m, n) satisfies the equation (m-1)(n-1) = 2 -/
def is_solution (m n : ℤ) : Prop := (m - 1) * (n - 1) = 2

theorem four_solutions :
  (∃ (S : Finset (ℤ × ℤ)), S.card = count_solutions ∧
    (∀ (p : ℤ × ℤ), p ∈ S ↔ is_solution p.1 p.2) ∧
    (∀ (m n : ℤ), is_solution m n → (m, n) ∈ S)) :=
sorry

end four_solutions_l2545_254570


namespace percentage_of_defective_meters_l2545_254512

theorem percentage_of_defective_meters
  (total_meters : ℕ)
  (rejected_meters : ℕ)
  (h1 : total_meters = 8000)
  (h2 : rejected_meters = 4) :
  (rejected_meters : ℝ) / (total_meters : ℝ) * 100 = 0.05 := by
  sorry

end percentage_of_defective_meters_l2545_254512


namespace total_ridges_l2545_254531

/-- The number of ridges on a single vinyl record -/
def ridges_per_record : ℕ := 60

/-- The number of cases Jerry has -/
def num_cases : ℕ := 4

/-- The number of shelves in each case -/
def shelves_per_case : ℕ := 3

/-- The number of records each shelf can hold -/
def records_per_shelf : ℕ := 20

/-- The percentage of shelf capacity that is full, represented as a rational number -/
def shelf_fullness : ℚ := 60 / 100

/-- Theorem stating the total number of ridges on Jerry's records -/
theorem total_ridges : 
  ridges_per_record * num_cases * shelves_per_case * records_per_shelf * shelf_fullness = 8640 := by
  sorry

end total_ridges_l2545_254531


namespace roots_equation_l2545_254561

theorem roots_equation (α β : ℝ) : 
  (α^2 - 3*α + 1 = 0) → 
  (β^2 - 3*β + 1 = 0) → 
  3*α^4 + 8*β^3 = 333 := by
  sorry

end roots_equation_l2545_254561


namespace trinomial_product_degree_15_l2545_254532

def trinomial (p q : ℕ) (a : ℝ) (x : ℝ) : ℝ := x^p + a * x^q + 1

theorem trinomial_product_degree_15 :
  ∀ (p q r s : ℕ) (a b : ℝ),
    q < p → s < r → p + r = 15 →
    (∃ (t : ℕ) (c : ℝ), 
      trinomial p q a * trinomial r s b = trinomial 15 t c) ↔
    ((p = 5 ∧ q = 0 ∧ r = 10 ∧ s = 5 ∧ a = 1 ∧ b = -1) ∨
     (p = 9 ∧ q = 3 ∧ r = 6 ∧ s = 3 ∧ a = -1 ∧ b = 1) ∨
     (p = 9 ∧ q = 6 ∧ r = 6 ∧ s = 3 ∧ a = -1 ∧ b = 1)) :=
by sorry

end trinomial_product_degree_15_l2545_254532


namespace gcd_multiple_equivalence_l2545_254554

theorem gcd_multiple_equivalence (d : ℕ) (h : d ≥ 1) :
  {m : ℕ | m ≥ 2 ∧ d ∣ m} =
  {m : ℕ | m ≥ 2 ∧ ∃ n : ℕ, n ≥ 1 ∧ Nat.gcd m n = d ∧ Nat.gcd m (4 * n + 1) = 1} :=
by sorry

end gcd_multiple_equivalence_l2545_254554


namespace average_age_when_youngest_born_l2545_254507

/-- Proves that for a group of 7 people with an average age of 30 and the youngest being 8 years old,
    the average age of the group when the youngest was born was 22 years. -/
theorem average_age_when_youngest_born
  (num_people : ℕ)
  (current_average_age : ℝ)
  (youngest_age : ℕ)
  (h_num_people : num_people = 7)
  (h_current_average : current_average_age = 30)
  (h_youngest : youngest_age = 8) :
  (num_people * current_average_age - num_people * youngest_age) / num_people = 22 :=
sorry

end average_age_when_youngest_born_l2545_254507


namespace monkey_arrangements_l2545_254537

theorem monkey_arrangements :
  (Finset.range 6).prod (λ i => 6 - i) = 720 := by
  sorry

end monkey_arrangements_l2545_254537


namespace equation_general_form_l2545_254573

theorem equation_general_form :
  ∀ x : ℝ, (x + 8) * (x - 1) = -5 ↔ x^2 + 7*x - 3 = 0 :=
by sorry

end equation_general_form_l2545_254573


namespace trevors_age_l2545_254510

theorem trevors_age (T : ℕ) : 
  (20 + (24 - T) = 3 * T) → T = 11 := by
  sorry

end trevors_age_l2545_254510


namespace valid_pairs_count_l2545_254538

/-- A function that checks if a positive integer has any zero digits -/
def has_zero_digit (n : ℕ+) : Bool :=
  sorry

/-- The set of positive integers less than or equal to 500 without zero digits -/
def valid_numbers : Set ℕ+ :=
  {n : ℕ+ | n ≤ 500 ∧ ¬(has_zero_digit n)}

/-- The number of ordered pairs (a, b) of positive integers where a + b = 500 
    and neither a nor b has a zero digit -/
def count_valid_pairs : ℕ :=
  sorry

theorem valid_pairs_count : count_valid_pairs = 93196 := by
  sorry

end valid_pairs_count_l2545_254538


namespace car_trip_average_mpg_l2545_254502

/-- Proves that the average miles per gallon for a car trip is 450/11, given specific conditions. -/
theorem car_trip_average_mpg :
  -- Define the distance from B to C as x
  ∀ x : ℝ,
  x > 0 →
  -- Distance from A to B is twice the distance from B to C
  let dist_ab := 2 * x
  let dist_bc := x
  -- Define the fuel efficiencies
  let mpg_ab := 25
  let mpg_bc := 30
  -- Calculate total distance and total fuel used
  let total_dist := dist_ab + dist_bc
  let total_fuel := dist_ab / mpg_ab + dist_bc / mpg_bc
  -- The average MPG for the entire trip
  let avg_mpg := total_dist / total_fuel
  -- Prove that the average MPG equals 450/11
  avg_mpg = 450 / 11 := by
    sorry

#eval (450 : ℚ) / 11

end car_trip_average_mpg_l2545_254502


namespace range_of_a_for_single_root_l2545_254523

-- Define the function f(x) = 2x³ - 3x² + a
def f (x a : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

-- State the theorem
theorem range_of_a_for_single_root :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ Set.Icc (-2) 2 ∧ f x a = 0) →
  a ∈ Set.Ioo (-4) 0 ∪ Set.Ioo 1 28 :=
sorry

end range_of_a_for_single_root_l2545_254523


namespace tank_fill_time_l2545_254509

/-- Represents a pipe with a flow rate (positive for filling, negative for draining) -/
structure Pipe where
  rate : Int

/-- Represents a tank with a capacity and a list of pipes -/
structure Tank where
  capacity : Nat
  pipes : List Pipe

def cycleTime : Nat := 3

def cycleVolume (tank : Tank) : Int :=
  tank.pipes.foldl (fun acc pipe => acc + pipe.rate) 0

theorem tank_fill_time (tank : Tank) (h1 : tank.capacity = 750)
    (h2 : tank.pipes = [⟨40⟩, ⟨30⟩, ⟨-20⟩])
    (h3 : cycleVolume tank = 50)
    (h4 : tank.capacity / cycleVolume tank * cycleTime = 45) :
  ∃ (t : Nat), t = 45 ∧ t * cycleVolume tank ≥ tank.capacity := by
  sorry

end tank_fill_time_l2545_254509


namespace millet_majority_on_sixth_day_l2545_254579

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  totalSeeds : ℚ
  milletSeeds : ℚ

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  let newTotalSeeds := state.totalSeeds + 2^(state.day - 1) / 2
  let newMilletSeeds := state.milletSeeds / 2 + 0.4 * 2^(state.day - 1) / 2
  { day := state.day + 1, totalSeeds := newTotalSeeds, milletSeeds := newMilletSeeds }

/-- The initial state of the feeder -/
def initialState : FeederState :=
  { day := 1, totalSeeds := 1/2, milletSeeds := 0.2 }

/-- Calculates the state of the feeder after n days -/
def stateAfterDays (n : Nat) : FeederState :=
  match n with
  | 0 => initialState
  | m + 1 => nextDay (stateAfterDays m)

/-- Theorem: On the 6th day, more than half of the seeds are millet -/
theorem millet_majority_on_sixth_day :
  let sixthDay := stateAfterDays 5
  sixthDay.milletSeeds > sixthDay.totalSeeds / 2 := by
  sorry

end millet_majority_on_sixth_day_l2545_254579


namespace triangle_inequality_l2545_254575

theorem triangle_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (hab : a + b ≥ c) (hbc : b + c ≥ a) (hca : c + a ≥ b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_l2545_254575


namespace imaginary_part_of_z_l2545_254558

theorem imaginary_part_of_z (z : ℂ) : z = ((Complex.I - 1)^2 + 4) / (Complex.I + 1) → z.im = -3 := by
  sorry

end imaginary_part_of_z_l2545_254558


namespace function_periodicity_l2545_254591

open Real

-- Define the function f and the constant a
variable (f : ℝ → ℝ) (a : ℝ)

-- State the theorem
theorem function_periodicity 
  (h : ∀ x, f (x + a) = (1 + f x) / (1 - f x)) 
  (ha : a ≠ 0) : 
  ∀ x, f (x + 4 * a) = f x := by
  sorry

end function_periodicity_l2545_254591


namespace geometric_sequence_constant_l2545_254590

/-- A geometric sequence with sum S_n = (a-2)⋅3^(n+1) + 2 -/
def GeometricSequence (a : ℝ) (n : ℕ) : ℝ := (a - 2) * 3^(n + 1) + 2

/-- The difference between consecutive sums gives the n-th term -/
def NthTerm (a : ℝ) (n : ℕ) : ℝ := GeometricSequence a n - GeometricSequence a (n - 1)

/-- Theorem stating that the constant a in the given geometric sequence is 4/3 -/
theorem geometric_sequence_constant : 
  ∃ (a : ℝ), (∀ n : ℕ, n ≥ 2 → (NthTerm a n) / (NthTerm a (n-1)) = (NthTerm a (n-1)) / (NthTerm a (n-2))) ∧ 
  a = 4/3 := by sorry

end geometric_sequence_constant_l2545_254590


namespace blue_paint_cans_l2545_254503

def blue_to_green_ratio : ℚ := 4 / 3
def total_cans : ℕ := 35

theorem blue_paint_cans : ℕ := by
  -- The number of cans of blue paint is 20
  sorry

end blue_paint_cans_l2545_254503


namespace simplify_fraction_cube_l2545_254535

theorem simplify_fraction_cube (a b : ℝ) (ha : a ≠ 0) :
  (3 * b / (2 * a^2))^3 = 27 * b^3 / (8 * a^6) := by sorry

end simplify_fraction_cube_l2545_254535


namespace worker_speed_comparison_l2545_254592

/-- Given that workers A and B can complete a work together in 18 days,
    and A alone can complete the work in 24 days,
    prove that A is 3 times faster than B. -/
theorem worker_speed_comparison (work : ℝ) (a_rate : ℝ) (b_rate : ℝ) :
  work > 0 →
  a_rate > 0 →
  b_rate > 0 →
  work / (a_rate + b_rate) = 18 →
  work / a_rate = 24 →
  a_rate / b_rate = 3 := by
  sorry

end worker_speed_comparison_l2545_254592


namespace pallet_weight_l2545_254572

/-- Given a pallet with 3 boxes, where each box weighs 89 kilograms,
    the total weight of the pallet is 267 kilograms. -/
theorem pallet_weight (num_boxes : ℕ) (weight_per_box : ℕ) (total_weight : ℕ) : 
  num_boxes = 3 → weight_per_box = 89 → total_weight = num_boxes * weight_per_box → 
  total_weight = 267 := by
  sorry

end pallet_weight_l2545_254572


namespace probability_of_target_letter_l2545_254548

def word : String := "CALCULATE"
def target_letters : String := "CUT"

def count_occurrences (c : Char) (s : String) : Nat :=
  s.toList.filter (· = c) |>.length

def favorable_outcomes : Nat :=
  target_letters.toList.map (λ c => count_occurrences c word) |>.sum

theorem probability_of_target_letter :
  (favorable_outcomes : ℚ) / word.length = 4 / 9 := by sorry

end probability_of_target_letter_l2545_254548


namespace point_in_fourth_quadrant_l2545_254571

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  fourth_quadrant 4 (-3) := by sorry

end point_in_fourth_quadrant_l2545_254571


namespace line_bisected_by_point_m_prove_line_bisected_by_point_m_l2545_254584

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a point is the midpoint of two other points -/
def Point.isMidpointOf (m : Point) (p1 p2 : Point) : Prop :=
  m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2

/-- The theorem to be proved -/
theorem line_bisected_by_point_m (l1 l2 : Line) (m : Point) : Prop :=
  let desired_line : Line := { a := 1, b := 4, c := -4 }
  let point_m : Point := { x := 0, y := 1 }
  l1 = { a := 1, b := -3, c := 10 } →
  l2 = { a := 2, b := 1, c := -8 } →
  m = point_m →
  ∃ (p1 p2 : Point),
    p1.liesOn l1 ∧
    p2.liesOn l2 ∧
    m.isMidpointOf p1 p2 ∧
    p1.liesOn desired_line ∧
    p2.liesOn desired_line ∧
    m.liesOn desired_line

/-- Proof of the theorem -/
theorem prove_line_bisected_by_point_m (l1 l2 : Line) (m : Point) :
  line_bisected_by_point_m l1 l2 m := by
  sorry

end line_bisected_by_point_m_prove_line_bisected_by_point_m_l2545_254584


namespace shopping_money_calculation_l2545_254518

theorem shopping_money_calculation (M : ℚ) : 
  (1 - 4/5 * (1 - 1/3 * (1 - 3/8))) * M = 1200 → M = 14400 := by
  sorry

end shopping_money_calculation_l2545_254518


namespace total_tickets_sold_l2545_254587

/-- Proves that the total number of tickets sold is 130 given the specified conditions -/
theorem total_tickets_sold (adult_price child_price total_receipts child_tickets : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_receipts = 840)
  (h4 : child_tickets = 90)
  : ∃ (adult_tickets : ℕ), adult_tickets * adult_price + child_tickets * child_price = total_receipts ∧ 
    adult_tickets + child_tickets = 130 := by
  sorry

end total_tickets_sold_l2545_254587


namespace no_integer_solution_l2545_254562

theorem no_integer_solution : ¬∃ (k l m n x : ℤ),
  (x = k * l * m * n) ∧
  (x - k = 1966) ∧
  (x - l = 966) ∧
  (x - m = 66) ∧
  (x - n = 6) := by
  sorry

end no_integer_solution_l2545_254562


namespace min_colors_l2545_254524

def is_divisor (a b : ℕ) : Prop := b % a = 0

def valid_coloring (f : ℕ → ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ 1000 ∧ 1 ≤ b ∧ b ≤ 1000 → 
    is_divisor a b → f a ≠ f b

theorem min_colors : 
  (∃ (n : ℕ) (f : ℕ → ℕ), n = 10 ∧ valid_coloring f ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 1000 → f i ≤ n)) ∧ 
  (∀ (m : ℕ) (g : ℕ → ℕ), m < 10 → 
    ¬(valid_coloring g ∧ (∀ i, 1 ≤ i ∧ i ≤ 1000 → g i ≤ m))) :=
by sorry

end min_colors_l2545_254524


namespace first_month_sale_proof_l2545_254547

/-- Calculates the sale in the first month given the sales for the next 5 months and the average sale for all 6 months. -/
def first_month_sale (month2 month3 month4 month5 month6 average : ℕ) : ℕ :=
  6 * average - (month2 + month3 + month4 + month5 + month6)

theorem first_month_sale_proof (month2 month3 month4 month5 month6 average : ℕ) :
  first_month_sale 6927 6855 7230 6562 7391 6900 = 6435 := by
  sorry

end first_month_sale_proof_l2545_254547


namespace ellipse_eccentricity_rectangle_l2545_254500

/-- Eccentricity of an ellipse with foci at opposite corners of a 4x3 rectangle 
    and passing through the other two corners -/
theorem ellipse_eccentricity_rectangle (a b c : ℝ) : 
  a = 4 →
  b = 3 →
  c = 2 →
  b^2 = 3*a →
  a^2 - c^2 = b^2 →
  c/a = 1/2 := by
  sorry

end ellipse_eccentricity_rectangle_l2545_254500


namespace arithmetic_sequence_sum_2_to_1000_l2545_254511

def arithmetic_sequence_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n : ℕ := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_2_to_1000 :
  arithmetic_sequence_sum 2 1000 2 = 250500 := by
  sorry

end arithmetic_sequence_sum_2_to_1000_l2545_254511


namespace smallest_a_value_l2545_254595

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b)
  (h3 : ∀ x : ℝ, Real.sin (a * x + b) = Real.sin (15 * x)) :
  a ≥ 15 ∧ ∀ a' ≥ 0, (∀ x : ℝ, Real.sin (a' * x + b) = Real.sin (15 * x)) → a' ≥ 15 :=
sorry

end smallest_a_value_l2545_254595


namespace shooting_game_equations_l2545_254598

/-- Represents the shooting game scenario -/
structure ShootingGame where
  x : ℕ  -- number of baskets Xiao Ming made
  y : ℕ  -- number of baskets his father made

/-- The conditions of the shooting game -/
def valid_game (g : ShootingGame) : Prop :=
  g.x + g.y = 20 ∧ 3 * g.x = g.y

theorem shooting_game_equations (g : ShootingGame) :
  valid_game g ↔ g.x + g.y = 20 ∧ 3 * g.x = g.y :=
sorry

end shooting_game_equations_l2545_254598


namespace solve_equation_l2545_254527

theorem solve_equation : ∃ a : ℝ, -2 - a = 0 ∧ a = -2 := by
  sorry

end solve_equation_l2545_254527


namespace cheryl_material_usage_l2545_254567

theorem cheryl_material_usage 
  (bought : ℚ) 
  (left : ℚ) 
  (h1 : bought = 3/8 + 1/3) 
  (h2 : left = 15/40) : 
  bought - left = 1/3 := by
sorry

end cheryl_material_usage_l2545_254567


namespace factorization_problems_l2545_254515

theorem factorization_problems (x : ℝ) : 
  (9 * x^2 - 6 * x + 1 = (3 * x - 1)^2) ∧ 
  (x^3 - x = x * (x + 1) * (x - 1)) := by
  sorry

end factorization_problems_l2545_254515


namespace cube_monotone_l2545_254556

theorem cube_monotone (a b : ℝ) : a > b → a^3 > b^3 := by sorry

end cube_monotone_l2545_254556


namespace cone_volume_given_sphere_l2545_254569

/-- Given a sphere and a cone with specific properties, prove that the volume of the cone is 12288π cm³ -/
theorem cone_volume_given_sphere (r_sphere : ℝ) (h_cone : ℝ) (r_cone : ℝ) :
  r_sphere = 24 →
  h_cone = 2 * r_sphere →
  π * r_cone * (r_cone + Real.sqrt (r_cone^2 + h_cone^2)) = 4 * π * r_sphere^2 →
  (1/3) * π * r_cone^2 * h_cone = 12288 * π := by
  sorry

#check cone_volume_given_sphere

end cone_volume_given_sphere_l2545_254569


namespace box_weights_sum_l2545_254545

theorem box_weights_sum (box1 box2 box3 box4 box5 : ℝ) 
  (h1 : box1 = 2.5)
  (h2 : box2 = 11.3)
  (h3 : box3 = 5.75)
  (h4 : box4 = 7.2)
  (h5 : box5 = 3.25) :
  box1 + box2 + box3 + box4 + box5 = 30 := by
  sorry

end box_weights_sum_l2545_254545


namespace larger_integer_value_l2545_254516

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  a = 21 := by
  sorry

end larger_integer_value_l2545_254516


namespace cereal_eating_time_l2545_254596

/-- The time it takes for Mr. Fat and Mr. Thin to eat 4 pounds of cereal together -/
def eating_time (fat_rate thin_rate total_cereal : ℚ) : ℕ :=
  (total_cereal / (fat_rate + thin_rate)).ceil.toNat

/-- Proves that Mr. Fat and Mr. Thin take 53 minutes to eat 4 pounds of cereal together -/
theorem cereal_eating_time :
  eating_time (1 / 20) (1 / 40) 4 = 53 := by
  sorry

end cereal_eating_time_l2545_254596


namespace solve_for_y_l2545_254533

theorem solve_for_y (t : ℚ) (x y : ℚ) 
  (hx : x = 3 - 2 * t) 
  (hy : y = 3 * t + 10) 
  (hx_val : x = -4) : 
  y = 41 / 2 := by
  sorry

end solve_for_y_l2545_254533


namespace arithmetic_sequence_sum_remainder_l2545_254526

/-- The sum of an arithmetic sequence with first term 3, last term 153, and common difference 5,
    when divided by 24, has a remainder of 0. -/
theorem arithmetic_sequence_sum_remainder (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 3 → aₙ = 153 → d = 5 → aₙ = a₁ + (n - 1) * d →
  (n * (a₁ + aₙ) / 2) % 24 = 0 :=
by sorry

end arithmetic_sequence_sum_remainder_l2545_254526


namespace folded_square_perimeter_l2545_254541

/-- A square with side length 2 is folded so that vertex A meets edge BC at A',
    and edge AB intersects edge CD at F. Given BA' = 1/2,
    prove that the perimeter of triangle CFA' is (3 + √17) / 2. -/
theorem folded_square_perimeter (A B C D A' F : ℝ × ℝ) : 
  let square_side : ℝ := 2
  let BA'_length : ℝ := 1/2
  -- Define the square
  (A = (0, square_side) ∧ B = (0, 0) ∧ C = (square_side, 0) ∧ D = (square_side, square_side)) →
  -- A' is on BC
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A' = (t * square_side, 0)) →
  -- BA' length is 1/2
  (Real.sqrt ((A'.1 - B.1)^2 + (A'.2 - B.2)^2) = BA'_length) →
  -- F is on CD
  (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F = (square_side, s * square_side)) →
  -- F is also on AB
  (∃ r : ℝ, 0 ≤ r ∧ r ≤ 1 ∧ F = ((1-r) * A.1 + r * B.1, (1-r) * A.2 + r * B.2)) →
  -- Conclusion: Perimeter of CFA' is (3 + √17) / 2
  let CF := Real.sqrt ((C.1 - F.1)^2 + (C.2 - F.2)^2)
  let FA' := Real.sqrt ((F.1 - A'.1)^2 + (F.2 - A'.2)^2)
  let CA' := Real.sqrt ((C.1 - A'.1)^2 + (C.2 - A'.2)^2)
  CF + FA' + CA' = (3 + Real.sqrt 17) / 2 := by
sorry

end folded_square_perimeter_l2545_254541


namespace digit_1257_of_7_19th_l2545_254599

/-- The decimal representation of 7/19 repeats every 18 digits -/
def period : ℕ := 18

/-- The repeating sequence of digits in the decimal representation of 7/19 -/
def repeating_sequence : List ℕ := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]

/-- The position we're interested in -/
def target_position : ℕ := 1257

/-- Theorem stating that the 1257th digit after the decimal point in 7/19 is 7 -/
theorem digit_1257_of_7_19th : 
  (repeating_sequence.get? ((target_position - 1) % period)) = some 7 := by
  sorry

end digit_1257_of_7_19th_l2545_254599


namespace probability_green_or_blue_ten_sided_die_l2545_254583

/-- Represents a 10-sided die with colored faces -/
structure ColoredDie :=
  (total_sides : Nat)
  (red_faces : Nat)
  (yellow_faces : Nat)
  (green_faces : Nat)
  (blue_faces : Nat)
  (valid_die : total_sides = red_faces + yellow_faces + green_faces + blue_faces)

/-- Calculates the probability of rolling either a green or blue face -/
def probability_green_or_blue (die : ColoredDie) : Rat :=
  (die.green_faces + die.blue_faces : Rat) / die.total_sides

/-- Theorem stating the probability of rolling either a green or blue face -/
theorem probability_green_or_blue_ten_sided_die :
  ∃ (die : ColoredDie),
    die.total_sides = 10 ∧
    die.red_faces = 4 ∧
    die.yellow_faces = 3 ∧
    die.green_faces = 2 ∧
    die.blue_faces = 1 ∧
    probability_green_or_blue die = 3 / 10 := by
  sorry

end probability_green_or_blue_ten_sided_die_l2545_254583


namespace thousandth_term_of_sequence_l2545_254543

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem thousandth_term_of_sequence :
  arithmetic_sequence 1 3 1000 = 2998 := by
  sorry

end thousandth_term_of_sequence_l2545_254543


namespace king_total_payment_l2545_254505

def crown_cost : ℚ := 20000
def architect_cost : ℚ := 50000
def chef_cost : ℚ := 10000

def crown_tip_percent : ℚ := 10 / 100
def architect_tip_percent : ℚ := 5 / 100
def chef_tip_percent : ℚ := 15 / 100

def total_cost : ℚ := crown_cost * (1 + crown_tip_percent) + 
                       architect_cost * (1 + architect_tip_percent) + 
                       chef_cost * (1 + chef_tip_percent)

theorem king_total_payment : total_cost = 86000 :=
by sorry

end king_total_payment_l2545_254505


namespace product_of_positive_real_solutions_l2545_254597

theorem product_of_positive_real_solutions (x : ℂ) : 
  (x^6 = -729) → 
  (∃ (S : Finset ℂ), 
    (∀ z ∈ S, z^6 = -729 ∧ z.re > 0) ∧ 
    (∀ z, z^6 = -729 ∧ z.re > 0 → z ∈ S) ∧
    (S.prod id = 9)) := by
sorry

end product_of_positive_real_solutions_l2545_254597


namespace rectangle_diagonal_l2545_254546

theorem rectangle_diagonal (l w : ℝ) (h_area : l * w = 20) (h_perimeter : 2 * l + 2 * w = 18) :
  l^2 + w^2 = 41 :=
sorry

end rectangle_diagonal_l2545_254546
