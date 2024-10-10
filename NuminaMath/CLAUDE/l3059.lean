import Mathlib

namespace third_number_in_ratio_l3059_305989

theorem third_number_in_ratio (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a : ℚ) / 5 = (b : ℚ) / 6 ∧ (b : ℚ) / 6 = (c : ℚ) / 8 →
  a + c = b + 49 →
  b = 42 := by sorry

end third_number_in_ratio_l3059_305989


namespace missing_number_is_four_l3059_305937

/-- The structure of the problem -/
structure BoxStructure where
  top_left : ℕ
  top_right : ℕ
  middle_left : ℕ
  middle_right : ℕ
  bottom : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (b : BoxStructure) : Prop :=
  b.middle_left = b.top_left * b.top_right ∧
  b.bottom = b.middle_left * b.middle_right ∧
  b.middle_left = 30 ∧
  b.top_left = 6 ∧
  b.top_right = 5 ∧
  b.bottom = 600

/-- The theorem to prove -/
theorem missing_number_is_four :
  ∀ b : BoxStructure, satisfies_conditions b → b.middle_right = 4 := by
  sorry

end missing_number_is_four_l3059_305937


namespace trig_problem_l3059_305953

theorem trig_problem (α β : Real) 
  (h1 : Real.cos (α - β/2) = -2 * Real.sqrt 7 / 7)
  (h2 : Real.sin (α/2 - β) = 1/2)
  (h3 : π/2 < α ∧ α < π)
  (h4 : 0 < β ∧ β < π/2) :
  Real.cos ((α + β)/2) = -Real.sqrt 21 / 14 ∧ 
  Real.tan (α + β) = 5 * Real.sqrt 3 / 11 := by
sorry

end trig_problem_l3059_305953


namespace stating_sock_drawing_probability_l3059_305922

/-- Represents the total number of socks -/
def total_socks : ℕ := 10

/-- Represents the number of colors -/
def num_colors : ℕ := 5

/-- Represents the number of socks per color -/
def socks_per_color : ℕ := 2

/-- Represents the number of socks drawn -/
def socks_drawn : ℕ := 5

/-- 
Theorem stating the probability of drawing 5 socks with exactly one pair 
of the same color and the rest different colors, given 10 socks with 
2 socks each of 5 colors.
-/
theorem sock_drawing_probability : 
  (total_socks = 10) → 
  (num_colors = 5) → 
  (socks_per_color = 2) → 
  (socks_drawn = 5) →
  (Prob_exactly_one_pair_rest_different : ℚ) →
  Prob_exactly_one_pair_rest_different = 10 / 63 := by
  sorry

end stating_sock_drawing_probability_l3059_305922


namespace perpendicular_parallel_transitive_l3059_305928

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_parallel_transitive 
  (a b c : Line) :
  perpendicular a b → parallel_line b c → perpendicular a c :=
by sorry

end perpendicular_parallel_transitive_l3059_305928


namespace power_calculation_l3059_305961

theorem power_calculation : 16^4 * 8^2 / 4^10 = 4 := by
  sorry

end power_calculation_l3059_305961


namespace polynomial_equality_l3059_305997

theorem polynomial_equality : 7^4 + 4*(7^3) + 6*(7^2) + 4*7 + 1 = 4096 := by
  sorry

end polynomial_equality_l3059_305997


namespace calculate_expression_l3059_305966

theorem calculate_expression : 3.75 - 1.267 + 0.48 = 2.963 := by
  sorry

end calculate_expression_l3059_305966


namespace halloween_bags_l3059_305990

theorem halloween_bags (total_students : ℕ) (pumpkin_students : ℕ) (pack_size : ℕ) (pack_price : ℕ) (individual_price : ℕ) (total_spent : ℕ) : 
  total_students = 25 →
  pumpkin_students = 14 →
  pack_size = 5 →
  pack_price = 3 →
  individual_price = 1 →
  total_spent = 17 →
  total_students - pumpkin_students = 11 :=
by sorry

end halloween_bags_l3059_305990


namespace correct_price_reduction_equation_l3059_305924

/-- Represents the price reduction scenario for a mobile phone -/
def price_reduction (original_price final_price x : ℝ) : Prop :=
  original_price * (1 - x)^2 = final_price

/-- Theorem stating the correct equation for the given price reduction scenario -/
theorem correct_price_reduction_equation :
  ∃ x : ℝ, price_reduction 1185 580 x :=
sorry

end correct_price_reduction_equation_l3059_305924


namespace sum_of_roots_equals_seven_l3059_305932

theorem sum_of_roots_equals_seven : ∃ (r₁ r₂ : ℝ), 
  r₁^2 - 7*r₁ + 10 = 0 ∧ 
  r₂^2 - 7*r₂ + 10 = 0 ∧ 
  r₁ + r₂ = 7 := by
  sorry

end sum_of_roots_equals_seven_l3059_305932


namespace hyperbola_equation_l3059_305910

/-- The equation of a hyperbola given specific conditions -/
theorem hyperbola_equation :
  ∀ (a b : ℝ) (P : ℝ × ℝ),
  a > 0 ∧ b > 0 →
  (∀ (x y : ℝ), y^2 = -8*x → (x + 2)^2 + y^2 = 4) →  -- Focus of parabola is (-2, 0)
  (P.1)^2 / a^2 - (P.2)^2 / b^2 = 1 →  -- P lies on the hyperbola
  P = (2 * Real.sqrt 3, 2) →
  (∀ (x y : ℝ), x^2 / 4 - y^2 / 2 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end hyperbola_equation_l3059_305910


namespace sprinting_competition_races_verify_sprinting_competition_races_l3059_305902

/-- Calculates the number of races needed to determine a champion in a sprinting competition. -/
def races_needed (total_sprinters : ℕ) (sprinters_per_race : ℕ) (eliminations_per_race : ℕ) : ℕ :=
  (total_sprinters - 1) / eliminations_per_race

/-- Theorem stating that 43 races are needed for the given competition setup. -/
theorem sprinting_competition_races : 
  races_needed 216 6 5 = 43 := by
  sorry

/-- Verifies the result by simulating rounds of the competition. -/
def verify_races (total_sprinters : ℕ) (sprinters_per_race : ℕ) : ℕ :=
  let first_round := total_sprinters / sprinters_per_race
  let second_round := first_round / sprinters_per_race
  let third_round := if second_round ≥ sprinters_per_race then 1 else 0
  first_round + second_round + third_round

/-- Theorem stating that the verification method also yields 43 races. -/
theorem verify_sprinting_competition_races :
  verify_races 216 6 = 43 := by
  sorry

end sprinting_competition_races_verify_sprinting_competition_races_l3059_305902


namespace tourists_travelers_checks_l3059_305931

/-- Represents the number of travelers checks of each denomination -/
structure TravelersChecks where
  fifty : Nat
  hundred : Nat

/-- The problem statement -/
theorem tourists_travelers_checks 
  (tc : TravelersChecks)
  (h1 : 50 * tc.fifty + 100 * tc.hundred = 1800)
  (h2 : tc.fifty ≥ 24)
  (h3 : (1800 - 50 * 24) / (tc.fifty + tc.hundred - 24) = 100) :
  tc.fifty + tc.hundred = 30 := by
  sorry

end tourists_travelers_checks_l3059_305931


namespace race_finish_times_l3059_305962

/-- Race parameters and runner speeds -/
def race_distance : ℝ := 15
def malcolm_speed : ℝ := 5
def joshua_speed : ℝ := 7
def emily_speed : ℝ := 6

/-- Calculate finish time for a runner given their speed -/
def finish_time (speed : ℝ) : ℝ := race_distance * speed

/-- Calculate time difference between two runners -/
def time_difference (speed1 speed2 : ℝ) : ℝ := finish_time speed1 - finish_time speed2

/-- Theorem stating the time differences for Joshua and Emily relative to Malcolm -/
theorem race_finish_times :
  (time_difference joshua_speed malcolm_speed = 30) ∧
  (time_difference emily_speed malcolm_speed = 15) := by
  sorry

end race_finish_times_l3059_305962


namespace base_eight_53_equals_43_l3059_305970

/-- Converts a two-digit base-eight number to base-ten. -/
def baseEightToBaseTen (n : Nat) : Nat :=
  let tens := n / 10
  let ones := n % 10
  tens * 8 + ones

/-- The base-eight number 53 is equal to 43 in base-ten. -/
theorem base_eight_53_equals_43 : baseEightToBaseTen 53 = 43 := by
  sorry

end base_eight_53_equals_43_l3059_305970


namespace circles_externally_tangent_l3059_305995

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2) = r1 + r2

theorem circles_externally_tangent :
  let c1 : ℝ × ℝ := (0, 0)
  let c2 : ℝ × ℝ := (3, 4)
  let r1 : ℝ := 2
  let r2 : ℝ := 3
  externally_tangent c1 c2 r1 r2 := by
  sorry

#check circles_externally_tangent

end circles_externally_tangent_l3059_305995


namespace odd_functions_properties_l3059_305945

-- Define the functions f and g
def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 + x - k
def g (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem odd_functions_properties :
  (∀ x, g (-x) = -g x) ∧  -- g is an odd function
  (g 1 = -2) ∧  -- g achieves minimum -2 at x = 1
  (∀ x, g x ≤ 2) ∧  -- maximum value of g is 2
  (∀ k, (∀ x ∈ Set.Icc (-1) 3, f k x ≤ g x) → k ≥ 8) ∧  -- range of k for f ≤ g on [-1,3]
  (∀ k, (∀ x₁ ∈ Set.Icc (-1) 3, ∀ x₂ ∈ Set.Icc (-1) 3, f k x₁ ≤ g x₂) → k ≥ 23)  -- range of k for f(x₁) ≤ g(x₂)
  := by sorry

end odd_functions_properties_l3059_305945


namespace negation_equivalence_l3059_305982

theorem negation_equivalence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) :=
by sorry

end negation_equivalence_l3059_305982


namespace cos_270_degrees_l3059_305927

theorem cos_270_degrees : Real.cos (270 * π / 180) = 0 := by
  sorry

end cos_270_degrees_l3059_305927


namespace eves_diner_purchase_l3059_305971

/-- The cost of a sandwich at Eve's Diner -/
def sandwich_cost : ℕ := 4

/-- The cost of a soda at Eve's Diner -/
def soda_cost : ℕ := 3

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 7

/-- The number of sodas purchased -/
def num_sodas : ℕ := 12

/-- The total cost of the purchase at Eve's Diner -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem eves_diner_purchase :
  total_cost = 64 := by sorry

end eves_diner_purchase_l3059_305971


namespace garrison_provision_days_l3059_305968

/-- The number of days provisions last for a garrison --/
def provisionDays (initialMen : ℕ) (reinforcementMen : ℕ) (daysBeforeReinforcement : ℕ) (daysAfterReinforcement : ℕ) : ℕ :=
  (initialMen * daysBeforeReinforcement + (initialMen + reinforcementMen) * daysAfterReinforcement) / initialMen

theorem garrison_provision_days :
  provisionDays 2000 2700 15 20 = 62 := by
  sorry

#eval provisionDays 2000 2700 15 20

end garrison_provision_days_l3059_305968


namespace tan_half_period_l3059_305929

/-- The period of tan(x/2) is 2π -/
theorem tan_half_period : 
  ∀ f : ℝ → ℝ, (∀ x, f x = Real.tan (x / 2)) → 
  ∃ p : ℝ, p > 0 ∧ (∀ x, f (x + p) = f x) ∧ p = 2 * Real.pi := by
  sorry

/-- The period of tan(x) is π -/
axiom tan_period : 
  ∀ x : ℝ, Real.tan (x + Real.pi) = Real.tan x

end tan_half_period_l3059_305929


namespace school_commute_time_l3059_305908

theorem school_commute_time (usual_rate : ℝ) (usual_time : ℝ) : 
  usual_time > 0 →
  usual_rate > 0 →
  (6 / 7 * usual_rate) * (usual_time - 4) = usual_rate * usual_time →
  usual_time = 28 := by
sorry

end school_commute_time_l3059_305908


namespace toothpaste_duration_l3059_305900

/-- Represents the amount of toothpaste in grams --/
def toothpasteAmount : ℝ := 105

/-- Represents the amount of toothpaste used by Anne's dad per brushing --/
def dadUsage : ℝ := 3

/-- Represents the amount of toothpaste used by Anne's mom per brushing --/
def momUsage : ℝ := 2

/-- Represents the amount of toothpaste used by Anne per brushing --/
def anneUsage : ℝ := 1

/-- Represents the amount of toothpaste used by Anne's brother per brushing --/
def brotherUsage : ℝ := 1

/-- Represents the number of times each family member brushes their teeth per day --/
def brushingsPerDay : ℕ := 3

/-- Theorem stating that the toothpaste will last for 5 days --/
theorem toothpaste_duration : 
  ∃ (days : ℝ), days = 5 ∧ 
  days * (dadUsage + momUsage + anneUsage + brotherUsage) * brushingsPerDay = toothpasteAmount :=
by sorry

end toothpaste_duration_l3059_305900


namespace trays_from_second_table_l3059_305980

def trays_per_trip : ℕ := 7
def total_trips : ℕ := 4
def trays_from_first_table : ℕ := 23

theorem trays_from_second_table :
  trays_per_trip * total_trips - trays_from_first_table = 5 := by
  sorry

end trays_from_second_table_l3059_305980


namespace problem_solution_l3059_305974

theorem problem_solution (m n : ℕ) 
  (h_pos_m : m > 0) 
  (h_pos_n : n > 0) 
  (h_inequality : m + 8 < n - 1) 
  (h_mean : (m + (m + 3) + (m + 8) + (n - 1) + (n + 3) + (2 * n - 2)) / 6 = n) 
  (h_median : (m + 8 + n - 1) / 2 = n) : 
  m + n = 47 := by
sorry

end problem_solution_l3059_305974


namespace cone_altitude_to_radius_ratio_l3059_305948

/-- The ratio of a cone's altitude to its base radius, given that its volume is one-third of a sphere with the same radius -/
theorem cone_altitude_to_radius_ratio (r h : ℝ) (h_pos : 0 < r) : 
  (1 / 3 * π * r^2 * h = 1 / 3 * (4 / 3 * π * r^3)) → h / r = 4 / 3 := by
  sorry

end cone_altitude_to_radius_ratio_l3059_305948


namespace element_in_set_l3059_305925

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : (U \ M) = {1, 3}) : 2 ∈ M := by
  sorry

end element_in_set_l3059_305925


namespace two_tangent_lines_l3059_305993

/-- The cubic function f(x) = -x³ + 6x² - 9x + 8 -/
def f (x : ℝ) : ℝ := -x^3 + 6*x^2 - 9*x + 8

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -3*x^2 + 12*x - 9

/-- Theorem: There are exactly two tangent lines from (0, 0) to the graph of f(x) -/
theorem two_tangent_lines :
  ∃! (s : Finset ℝ), s.card = 2 ∧
    ∀ x₀ ∈ s, f x₀ + f' x₀ * (-x₀) = 0 ∧
    ∀ x ∉ s, f x + f' x * (-x) ≠ 0 :=
sorry

end two_tangent_lines_l3059_305993


namespace no_real_solutions_l3059_305920

theorem no_real_solutions : ¬∃ (x : ℝ), x > 0 ∧ x^(1/4) = 15 / (8 - 2 * x^(1/4)) := by
  sorry

end no_real_solutions_l3059_305920


namespace parallel_planes_from_perpendicular_lines_l3059_305985

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planes_parallel : Plane → Plane → Prop)
variable (non_intersecting_lines : Line → Line → Prop)
variable (non_intersecting_planes : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_from_perpendicular_lines 
  (m n : Line) (α β : Plane) : 
  non_intersecting_lines m n →
  non_intersecting_planes α β →
  parallel m n →
  perpendicular m α →
  perpendicular n β →
  planes_parallel α β :=
sorry

end parallel_planes_from_perpendicular_lines_l3059_305985


namespace smallest_prime_scalene_perimeter_l3059_305954

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if three numbers form a scalene triangle -/
def isScaleneTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- The theorem stating the smallest possible perimeter of a scalene triangle with prime side lengths -/
theorem smallest_prime_scalene_perimeter :
  ∀ a b c : ℕ,
    isPrime a → isPrime b → isPrime c →
    isScaleneTriangle a b c →
    a + b + c ≥ 15 :=
by sorry

end smallest_prime_scalene_perimeter_l3059_305954


namespace smallest_positive_k_l3059_305969

theorem smallest_positive_k (m n : ℕ+) (h : m ≤ 2000) : 
  let k := 3 - (m : ℚ) / n
  ∀ k' > 0, k ≥ k' → k' ≥ 1/667 :=
by sorry

end smallest_positive_k_l3059_305969


namespace test_questions_count_l3059_305933

theorem test_questions_count : 
  ∀ (total : ℕ), 
    (total % 4 = 0) →  -- The test has 4 sections with equal number of questions
    (20 : ℚ) / total > (60 : ℚ) / 100 → -- Correct answer percentage > 60%
    (20 : ℚ) / total < (70 : ℚ) / 100 → -- Correct answer percentage < 70%
    total = 32 := by
  sorry

end test_questions_count_l3059_305933


namespace senior_ticket_price_l3059_305951

/-- Represents the cost of movie tickets for a family --/
structure MovieTickets where
  adult_price : ℕ
  child_price : ℕ
  total_cost : ℕ
  num_adults : ℕ
  num_children : ℕ
  num_seniors : ℕ

/-- Theorem stating the price of a senior citizen's ticket --/
theorem senior_ticket_price (tickets : MovieTickets) 
  (h1 : tickets.adult_price = 11)
  (h2 : tickets.child_price = 8)
  (h3 : tickets.total_cost = 64)
  (h4 : tickets.num_adults = 3)
  (h5 : tickets.num_children = 2)
  (h6 : tickets.num_seniors = 2) :
  tickets.total_cost = 
    tickets.num_adults * tickets.adult_price + 
    tickets.num_children * tickets.child_price + 
    tickets.num_seniors * 13 :=
by sorry

end senior_ticket_price_l3059_305951


namespace total_length_eleven_segments_l3059_305956

/-- The total length of 11 congruent segments -/
def total_length (segment_length : ℝ) (num_segments : ℕ) : ℝ :=
  segment_length * (num_segments : ℝ)

/-- Theorem: The total length of 11 congruent segments of 7 cm each is 77 cm -/
theorem total_length_eleven_segments :
  total_length 7 11 = 77 := by sorry

end total_length_eleven_segments_l3059_305956


namespace sqrt_450_simplification_l3059_305918

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplification_l3059_305918


namespace arithmetic_sequence_logarithm_l3059_305901

theorem arithmetic_sequence_logarithm (a b : ℝ) (m : ℝ) :
  a > 0 ∧ b > 0 ∧
  (2 : ℝ) ^ a = m ∧
  (3 : ℝ) ^ b = m ∧
  2 * a * b = a + b →
  m = Real.sqrt 6 := by
sorry

end arithmetic_sequence_logarithm_l3059_305901


namespace shirt_price_calculation_l3059_305986

theorem shirt_price_calculation (total_cost sweater_price shirt_price : ℝ) :
  total_cost = 80.34 →
  shirt_price = sweater_price - 7.43 →
  total_cost = sweater_price + shirt_price →
  shirt_price = 36.455 := by
sorry

end shirt_price_calculation_l3059_305986


namespace sundress_price_problem_l3059_305917

theorem sundress_price_problem (P : ℝ) : 
  P - (P * 0.85 * 1.25) = 4.5 → P * 0.85 = 61.2 := by
  sorry

end sundress_price_problem_l3059_305917


namespace circle_area_l3059_305905

theorem circle_area (r : ℝ) (h : 6 * (1 / (2 * Real.pi * r)) = r) : π * r^2 = 3 := by
  sorry

end circle_area_l3059_305905


namespace smaller_number_problem_l3059_305958

theorem smaller_number_problem (x y : ℤ) 
  (h1 : x = 2 * y - 3) 
  (h2 : x + y = 51) : 
  min x y = 18 := by sorry

end smaller_number_problem_l3059_305958


namespace function_extrema_sum_l3059_305911

/-- Given f(x) = 2x^3 - ax^2 + 1 where a > 0, if the sum of the maximum and minimum values 
    of f(x) on [-1, 1] is 1, then a = 1/2 -/
theorem function_extrema_sum (a : ℝ) (h1 : a > 0) : 
  let f := fun x => 2 * x^3 - a * x^2 + 1
  (∃ M m : ℝ, (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ M ∧ m ≤ f x) ∧ M + m = 1) → 
  a = 1/2 := by
sorry

end function_extrema_sum_l3059_305911


namespace sum_mod_thirteen_l3059_305904

theorem sum_mod_thirteen : (9010 + 9011 + 9012 + 9013 + 9014) % 13 = 9 := by
  sorry

end sum_mod_thirteen_l3059_305904


namespace rational_function_value_l3059_305991

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_linear : ∃ a b : ℝ, ∀ x, p x = a * x + b
  q_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  asymptote_neg_four : q (-4) = 0
  asymptote_one : q 1 = 0
  passes_origin : p 0 = 0 ∧ q 0 ≠ 0
  passes_two_neg_one : p 2 / q 2 = -1

/-- The main theorem -/
theorem rational_function_value (f : RationalFunction) : f.p 1 / f.q 1 = -3/5 := by
  sorry

end rational_function_value_l3059_305991


namespace expression_equals_three_l3059_305946

theorem expression_equals_three :
  |Real.sqrt 3 - 1| + (2023 - Real.pi)^0 - (-1/3)⁻¹ - 3 * Real.tan (30 * π / 180) = 3 := by
  sorry

end expression_equals_three_l3059_305946


namespace not_always_divisible_by_19_l3059_305934

theorem not_always_divisible_by_19 : ∃ (a b : ℤ), ¬(19 ∣ ((3*a + 2)^3 - (3*b + 2)^3)) := by
  sorry

end not_always_divisible_by_19_l3059_305934


namespace basketball_teams_l3059_305998

theorem basketball_teams (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end basketball_teams_l3059_305998


namespace unique_a_value_l3059_305942

def A (a : ℝ) : Set ℝ := {0, 2, a^2}
def B (a : ℝ) : Set ℝ := {1, a}

theorem unique_a_value : ∃! a : ℝ, A a ∪ B a = {0, 1, 2, 4} := by
  sorry

end unique_a_value_l3059_305942


namespace dining_bill_calculation_l3059_305959

theorem dining_bill_calculation (total : ℝ) (tip_rate : ℝ) (tax_rate : ℝ) 
  (h1 : total = 132)
  (h2 : tip_rate = 0.20)
  (h3 : tax_rate = 0.10) :
  ∃ (original_price : ℝ), 
    original_price * (1 + tax_rate) * (1 + tip_rate) = total ∧ 
    original_price = 100 := by
  sorry

end dining_bill_calculation_l3059_305959


namespace arcade_candy_cost_l3059_305907

theorem arcade_candy_cost (tickets_game1 tickets_game2 candies : ℕ) 
  (h1 : tickets_game1 = 33)
  (h2 : tickets_game2 = 9)
  (h3 : candies = 7) :
  (tickets_game1 + tickets_game2) / candies = 6 :=
by sorry

end arcade_candy_cost_l3059_305907


namespace xiaoming_age_proof_l3059_305909

/-- Xiaoming's current age -/
def xiaoming_age : ℕ := 6

/-- The current age of each of Xiaoming's younger brothers -/
def brother_age : ℕ := 2

/-- The number of Xiaoming's younger brothers -/
def num_brothers : ℕ := 3

/-- Years into the future for the second condition -/
def future_years : ℕ := 6

theorem xiaoming_age_proof :
  (xiaoming_age = num_brothers * brother_age) ∧
  (num_brothers * (brother_age + future_years) = 2 * (xiaoming_age + future_years)) →
  xiaoming_age = 6 := by
  sorry

end xiaoming_age_proof_l3059_305909


namespace partner_a_share_l3059_305983

/-- Calculates a partner's share of the annual gain in a partnership --/
def calculate_share (x : ℚ) (annual_gain : ℚ) : ℚ :=
  let a_share := 12 * x
  let b_share := 12 * x
  let c_share := 12 * x
  let d_share := 36 * x
  let e_share := 35 * x
  let f_share := 30 * x
  let total_investment := a_share + b_share + c_share + d_share + e_share + f_share
  (a_share / total_investment) * annual_gain

/-- The problem statement --/
theorem partner_a_share :
  ∃ (x : ℚ), calculate_share x 38400 = 3360 := by
  sorry

end partner_a_share_l3059_305983


namespace always_unaffected_square_l3059_305939

/-- Represents a square cut on the cake -/
structure Cut where
  x : ℚ
  y : ℚ
  size : ℚ
  h_x : 0 ≤ x ∧ x + size ≤ 3
  h_y : 0 ≤ y ∧ y + size ≤ 3

/-- Represents a small 1/3 x 1/3 square on the cake -/
structure SmallSquare where
  x : ℚ
  y : ℚ
  h_x : x = 0 ∨ x = 1 ∨ x = 2
  h_y : y = 0 ∨ y = 1 ∨ y = 2

/-- Check if a small square is affected by a cut -/
def isAffected (s : SmallSquare) (c : Cut) : Prop :=
  (c.x < s.x + 1/3 ∧ s.x < c.x + c.size) ∧
  (c.y < s.y + 1/3 ∧ s.y < c.y + c.size)

/-- Main theorem: There always exists an unaffected 1/3 x 1/3 square -/
theorem always_unaffected_square (cuts : Finset Cut) (h : cuts.card = 4) (h_size : ∀ c ∈ cuts, c.size = 1) :
  ∃ s : SmallSquare, ∀ c ∈ cuts, ¬isAffected s c :=
sorry

end always_unaffected_square_l3059_305939


namespace nickel_probability_l3059_305955

/-- Represents the types of coins in the box -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in the box in cents -/
def total_value : Coin → ℕ
  | Coin.Dime => 500
  | Coin.Nickel => 250
  | Coin.Penny => 100

/-- The number of coins of each type in the box -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the box -/
def total_coins : ℕ := coin_count Coin.Dime + coin_count Coin.Nickel + coin_count Coin.Penny

/-- The probability of randomly selecting a nickel from the box -/
theorem nickel_probability : 
  (coin_count Coin.Nickel : ℚ) / total_coins = 1 / 4 := by sorry

end nickel_probability_l3059_305955


namespace unique_solution_condition_l3059_305903

theorem unique_solution_condition (t : ℝ) :
  (∃! x y z v : ℝ, x + y + z + v = 0 ∧ (x*y + y*z + z*v) + t*(x*z + x*v + y*v) = 0) ↔
  (t > (3 - Real.sqrt 5) / 2 ∧ t < (3 + Real.sqrt 5) / 2) :=
by sorry

end unique_solution_condition_l3059_305903


namespace max_value_even_function_l3059_305981

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem max_value_even_function 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_max : ∃ x ∈ Set.Icc (-3) (-1), ∀ y ∈ Set.Icc (-3) (-1), f y ≤ f x ∧ f x = 6) :
  ∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, f y ≤ f x ∧ f x = 6 :=
sorry

end max_value_even_function_l3059_305981


namespace even_function_properties_l3059_305964

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem even_function_properties (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_sum_zero : ∀ x, f x + f (2 - x) = 0) :
  is_periodic f 4 ∧ is_odd (fun x ↦ f (x - 1)) := by
  sorry

end even_function_properties_l3059_305964


namespace paper_remaining_l3059_305944

theorem paper_remaining (total : ℕ) (used : ℕ) (h1 : total = 900) (h2 : used = 156) :
  total - used = 744 := by
  sorry

end paper_remaining_l3059_305944


namespace product_of_square_roots_l3059_305994

theorem product_of_square_roots (x : ℝ) (h : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (8 * x) = 60 * x * Real.sqrt x := by
  sorry

end product_of_square_roots_l3059_305994


namespace gadget_sales_sum_l3059_305930

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Gadget sales problem -/
theorem gadget_sales_sum :
  arithmetic_sum 2 3 25 = 950 := by
  sorry

end gadget_sales_sum_l3059_305930


namespace unique_w_value_l3059_305992

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ := sorry

def consecutive_digit_sums_prime (n : ℕ) : Prop := sorry

theorem unique_w_value (w : ℕ) :
  w > 0 →
  digit_sum (10^w - 74) = 440 →
  consecutive_digit_sums_prime (10^w - 74) →
  w = 50 := by sorry

end unique_w_value_l3059_305992


namespace total_non_hot_peppers_l3059_305977

/-- Represents the days of the week -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the pepper subtypes -/
inductive PepperSubtype
  | Jalapeno
  | Habanero
  | Bell
  | Banana
  | Poblano
  | Anaheim

/-- Returns the number of peppers picked for a given subtype on a given day -/
def peppers_picked (day : Day) (subtype : PepperSubtype) : Nat :=
  match day, subtype with
  | Day.Sunday,    PepperSubtype.Jalapeno  => 3
  | Day.Sunday,    PepperSubtype.Habanero  => 4
  | Day.Sunday,    PepperSubtype.Bell      => 6
  | Day.Sunday,    PepperSubtype.Banana    => 4
  | Day.Sunday,    PepperSubtype.Poblano   => 7
  | Day.Sunday,    PepperSubtype.Anaheim   => 6
  | Day.Monday,    PepperSubtype.Jalapeno  => 6
  | Day.Monday,    PepperSubtype.Habanero  => 6
  | Day.Monday,    PepperSubtype.Bell      => 4
  | Day.Monday,    PepperSubtype.Banana    => 4
  | Day.Monday,    PepperSubtype.Poblano   => 5
  | Day.Monday,    PepperSubtype.Anaheim   => 5
  | Day.Tuesday,   PepperSubtype.Jalapeno  => 7
  | Day.Tuesday,   PepperSubtype.Habanero  => 7
  | Day.Tuesday,   PepperSubtype.Bell      => 10
  | Day.Tuesday,   PepperSubtype.Banana    => 9
  | Day.Tuesday,   PepperSubtype.Poblano   => 4
  | Day.Tuesday,   PepperSubtype.Anaheim   => 3
  | Day.Wednesday, PepperSubtype.Jalapeno  => 6
  | Day.Wednesday, PepperSubtype.Habanero  => 6
  | Day.Wednesday, PepperSubtype.Bell      => 3
  | Day.Wednesday, PepperSubtype.Banana    => 2
  | Day.Wednesday, PepperSubtype.Poblano   => 12
  | Day.Wednesday, PepperSubtype.Anaheim   => 11
  | Day.Thursday,  PepperSubtype.Jalapeno  => 3
  | Day.Thursday,  PepperSubtype.Habanero  => 2
  | Day.Thursday,  PepperSubtype.Bell      => 10
  | Day.Thursday,  PepperSubtype.Banana    => 10
  | Day.Thursday,  PepperSubtype.Poblano   => 3
  | Day.Thursday,  PepperSubtype.Anaheim   => 2
  | Day.Friday,    PepperSubtype.Jalapeno  => 9
  | Day.Friday,    PepperSubtype.Habanero  => 9
  | Day.Friday,    PepperSubtype.Bell      => 8
  | Day.Friday,    PepperSubtype.Banana    => 7
  | Day.Friday,    PepperSubtype.Poblano   => 6
  | Day.Friday,    PepperSubtype.Anaheim   => 6
  | Day.Saturday,  PepperSubtype.Jalapeno  => 6
  | Day.Saturday,  PepperSubtype.Habanero  => 6
  | Day.Saturday,  PepperSubtype.Bell      => 4
  | Day.Saturday,  PepperSubtype.Banana    => 4
  | Day.Saturday,  PepperSubtype.Poblano   => 15
  | Day.Saturday,  PepperSubtype.Anaheim   => 15

/-- Returns true if the pepper subtype is non-hot (sweet or mild) -/
def is_non_hot (subtype : PepperSubtype) : Bool :=
  match subtype with
  | PepperSubtype.Bell    => true
  | PepperSubtype.Banana  => true
  | PepperSubtype.Poblano => true
  | PepperSubtype.Anaheim => true
  | _                     => false

/-- Theorem: The total number of non-hot peppers picked throughout the week is 185 -/
theorem total_non_hot_peppers :
  (List.sum (List.map
    (fun day =>
      List.sum (List.map
        (fun subtype =>
          if is_non_hot subtype then peppers_picked day subtype else 0)
        [PepperSubtype.Jalapeno, PepperSubtype.Habanero, PepperSubtype.Bell,
         PepperSubtype.Banana, PepperSubtype.Poblano, PepperSubtype.Anaheim]))
    [Day.Sunday, Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday, Day.Saturday]))
  = 185 := by
  sorry

end total_non_hot_peppers_l3059_305977


namespace inverse_proportion_points_order_l3059_305963

theorem inverse_proportion_points_order (x₁ x₂ x₃ : ℝ) : 
  10 / x₁ = -5 → 10 / x₂ = 2 → 10 / x₃ = 5 → x₁ < x₃ ∧ x₃ < x₂ := by
  sorry

end inverse_proportion_points_order_l3059_305963


namespace drama_club_ticket_sales_l3059_305973

theorem drama_club_ticket_sales 
  (total_tickets : ℕ) 
  (adult_price student_price : ℚ) 
  (total_amount : ℚ) 
  (h1 : total_tickets = 1500)
  (h2 : adult_price = 12)
  (h3 : student_price = 6)
  (h4 : total_amount = 16200) :
  ∃ (adult_tickets student_tickets : ℕ),
    adult_tickets + student_tickets = total_tickets ∧
    adult_price * adult_tickets + student_price * student_tickets = total_amount ∧
    student_tickets = 300 := by
  sorry

end drama_club_ticket_sales_l3059_305973


namespace wire_cutting_l3059_305996

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 70 →
  ratio = 2 / 3 →
  shorter_piece + ratio * shorter_piece = total_length →
  shorter_piece = 42 := by
sorry

end wire_cutting_l3059_305996


namespace intersection_line_slope_l3059_305979

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 9 = 0

-- Define the intersection points
def intersection (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 C.1 C.2 ∧ circle2 D.1 D.2 ∧ C ≠ D

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) (h : intersection C D) :
  (D.2 - C.2) / (D.1 - C.1) = -1 := by sorry

end intersection_line_slope_l3059_305979


namespace larger_solution_of_quadratic_l3059_305984

theorem larger_solution_of_quadratic (x : ℝ) :
  x^2 - 13*x + 30 = 0 ∧ x ≠ 3 → x = 10 :=
by
  sorry

end larger_solution_of_quadratic_l3059_305984


namespace smallest_survey_size_l3059_305957

theorem smallest_survey_size : ∀ n : ℕ, 
  n > 0 → 
  (∃ y n_yes n_no : ℕ, 
    n_yes = (76 * n) / 100 ∧ 
    n_no = (24 * n) / 100 ∧ 
    n_yes + n_no = n) → 
  n ≥ 25 := by sorry

end smallest_survey_size_l3059_305957


namespace geometric_sequence_min_a1_l3059_305952

theorem geometric_sequence_min_a1 (a : ℕ+ → ℕ+) (r : ℕ+) :
  (∀ i : ℕ+, a (i + 1) = a i * r) →  -- Geometric sequence condition
  (a 20 + a 21 = 20^21) →            -- Given condition
  (∃ x y : ℕ+, (∀ k : ℕ+, a 1 ≤ 2^(x:ℕ) * 5^(y:ℕ)) ∧ 
               a 1 = 2^(x:ℕ) * 5^(y:ℕ) ∧ 
               x + y = 24) :=
by sorry

end geometric_sequence_min_a1_l3059_305952


namespace product_of_base8_digits_9876_l3059_305976

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

theorem product_of_base8_digits_9876 :
  productOfList (toBase8 9876) = 96 :=
by sorry

end product_of_base8_digits_9876_l3059_305976


namespace weekly_reading_time_l3059_305936

def daily_meditation_time : ℝ := 1
def daily_reading_time : ℝ := 2 * daily_meditation_time
def days_in_week : ℕ := 7

theorem weekly_reading_time :
  daily_reading_time * (days_in_week : ℝ) = 14 := by
  sorry

end weekly_reading_time_l3059_305936


namespace circle_radius_is_4_l3059_305949

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + y^2 + 4*y + 13 = 0

-- Define the radius of the circle
def circle_radius : ℝ := 4

-- Theorem statement
theorem circle_radius_is_4 :
  ∀ x y : ℝ, circle_equation x y → 
  ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end circle_radius_is_4_l3059_305949


namespace correct_marble_distribution_l3059_305916

/-- Represents the number of marbles each boy has -/
structure MarbleDistribution where
  middle : ℕ
  least : ℕ
  most : ℕ

/-- Checks if the given marble distribution satisfies the problem conditions -/
def is_valid_distribution (d : MarbleDistribution) : Prop :=
  -- The ratio of marbles is 4:2:3
  4 * d.middle = 2 * d.most ∧
  2 * d.least = 3 * d.middle ∧
  -- The boy with the least marbles has 10 more than twice the middle boy's marbles
  d.least = 2 * d.middle + 10 ∧
  -- The total number of marbles is 156
  d.middle + d.least + d.most = 156

/-- The theorem stating the correct distribution of marbles -/
theorem correct_marble_distribution :
  is_valid_distribution ⟨23, 57, 76⟩ := by sorry

end correct_marble_distribution_l3059_305916


namespace plane_sphere_ratio_sum_l3059_305913

/-- Given a plane passing through (a,b,c) and intersecting the coordinate axes, 
    prove that the sum of ratios of the fixed point coordinates to the sphere center coordinates is 2. -/
theorem plane_sphere_ratio_sum (a b c d e f p q r : ℝ) 
  (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)
  (hdist : d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0)
  (hplane : a / d + b / e + c / f = 1)
  (hsphere : p^2 + q^2 + r^2 = (p - d)^2 + q^2 + r^2 ∧ 
             p^2 + q^2 + r^2 = p^2 + (q - e)^2 + r^2 ∧ 
             p^2 + q^2 + r^2 = p^2 + q^2 + (r - f)^2) :
  a / p + b / q + c / r = 2 := by
sorry

end plane_sphere_ratio_sum_l3059_305913


namespace smallest_collection_l3059_305960

def yoongi_collection : ℕ := 4
def yuna_collection : ℕ := 5
def jungkook_collection : ℕ := 6 + 3

theorem smallest_collection : 
  yoongi_collection < yuna_collection ∧ 
  yoongi_collection < jungkook_collection := by
sorry

end smallest_collection_l3059_305960


namespace parabola_above_line_l3059_305914

theorem parabola_above_line (a : ℝ) : 
  (∀ x ∈ Set.Icc a (a + 1), x^2 - a*x + 3 > 9/4) ↔ a > -Real.sqrt 3 := by
  sorry

end parabola_above_line_l3059_305914


namespace notification_completeness_l3059_305987

/-- Represents a point in the kingdom --/
structure Point where
  x : Real
  y : Real

/-- Represents the kingdom --/
structure Kingdom where
  side_length : Real
  residents : Set Point

/-- Represents the notification process --/
def NotificationProcess (k : Kingdom) (speed : Real) (start_time : Real) (end_time : Real) :=
  ∀ p ∈ k.residents, ∃ t : Real, start_time ≤ t ∧ t ≤ end_time ∧
    ∃ q : Point, q ∈ k.residents ∧ 
    Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2) ≤ speed * (t - start_time)

theorem notification_completeness 
  (k : Kingdom) 
  (h1 : k.side_length = 2) 
  (h2 : ∀ p ∈ k.residents, 0 ≤ p.x ∧ p.x ≤ k.side_length ∧ 0 ≤ p.y ∧ p.y ≤ k.side_length)
  (speed : Real) 
  (h3 : speed = 3) 
  (start_time end_time : Real) 
  (h4 : start_time = 12) 
  (h5 : end_time = 18) :
  NotificationProcess k speed start_time end_time :=
sorry

end notification_completeness_l3059_305987


namespace x_intercept_of_line_l3059_305965

/-- The x-intercept of the line 3x + 5y = 20 is (20/3, 0) -/
theorem x_intercept_of_line (x y : ℚ) : 
  3 * x + 5 * y = 20 → y = 0 → x = 20 / 3 := by
  sorry

end x_intercept_of_line_l3059_305965


namespace nonzero_real_equation_solution_l3059_305919

theorem nonzero_real_equation_solution (x : ℝ) (h : x ≠ 0) :
  (5 * x)^10 = (10 * x)^5 ↔ x = 2/5 := by sorry

end nonzero_real_equation_solution_l3059_305919


namespace expression_one_equals_negative_one_expression_two_equals_five_l3059_305999

-- Expression 1
theorem expression_one_equals_negative_one :
  (9/4)^(1/2) - (-8.6)^0 - (8/27)^(-1/3) = -1 := by sorry

-- Expression 2
theorem expression_two_equals_five :
  Real.log 25 / Real.log 10 + Real.log 4 / Real.log 10 + 7^(Real.log 2 / Real.log 7) + 2 * (Real.log 3 / (2 * Real.log 3)) = 5 := by sorry

end expression_one_equals_negative_one_expression_two_equals_five_l3059_305999


namespace geometric_sequence_ratio_l3059_305947

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_ratio
  (a : ℕ → ℚ)
  (h_geometric : geometric_sequence a)
  (h_a2 : a 2 = 2)
  (h_a6 : a 6 = 1/8) :
  ∃ q : ℚ, (q = 1/2 ∨ q = -1/2) ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end geometric_sequence_ratio_l3059_305947


namespace mustard_bottles_sum_l3059_305935

theorem mustard_bottles_sum : 0.25 + 0.25 + 0.38 = 0.88 := by
  sorry

end mustard_bottles_sum_l3059_305935


namespace inequality_proof_l3059_305978

-- Define the logarithm function with base 1/8
noncomputable def log_base_1_8 (x : ℝ) : ℝ := Real.log x / Real.log (1/8)

-- State the theorem
theorem inequality_proof (x : ℝ) (h1 : x ≥ 1/2) (h2 : x < 1) :
  9.244 * Real.sqrt (1 - 9 * (log_base_1_8 x)^2) > 1 - 4 * log_base_1_8 x :=
by sorry

end inequality_proof_l3059_305978


namespace remaining_area_after_triangles_cut_l3059_305912

theorem remaining_area_after_triangles_cut (grid_side : ℕ) (dark_rect_dim : ℕ × ℕ) (light_rect_dim : ℕ × ℕ) : 
  grid_side = 6 →
  dark_rect_dim = (1, 3) →
  light_rect_dim = (2, 3) →
  (grid_side^2 : ℝ) - (dark_rect_dim.1 * dark_rect_dim.2 + light_rect_dim.1 * light_rect_dim.2 : ℝ) = 27 := by
  sorry

end remaining_area_after_triangles_cut_l3059_305912


namespace cos_pi_third_plus_two_alpha_l3059_305943

theorem cos_pi_third_plus_two_alpha (α : Real) 
  (h : Real.sin (π / 3 - α) = 1 / 4) : 
  Real.cos (π / 3 + 2 * α) = -7 / 8 := by
  sorry

end cos_pi_third_plus_two_alpha_l3059_305943


namespace f_constant_iff_max_value_expression_exists_max_value_expression_l3059_305967

-- Part 1
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem f_constant_iff (x : ℝ) : (∀ y ∈ Set.Icc (-3) 1, f y = f x) ↔ x ∈ Set.Icc (-3) 1 := by sorry

-- Part 2
theorem max_value_expression (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  Real.sqrt 2 * x + Real.sqrt 2 * y + Real.sqrt 5 * z ≤ 3 := by sorry

theorem exists_max_value_expression :
  ∃ x y z : ℝ, x^2 + y^2 + z^2 = 1 ∧ Real.sqrt 2 * x + Real.sqrt 2 * y + Real.sqrt 5 * z = 3 := by sorry

end f_constant_iff_max_value_expression_exists_max_value_expression_l3059_305967


namespace correct_operation_l3059_305975

theorem correct_operation (x : ℤ) : (x - 7) * 20 = -380 → (x * 7) - 20 = -104 := by
  sorry

end correct_operation_l3059_305975


namespace perpendicular_line_equation_l3059_305940

/-- The equation of the line passing through the center of the circle x^2 + 2x + y^2 = 0
    and perpendicular to the line x + y = 0 is x - y + 1 = 0. -/
theorem perpendicular_line_equation : ∃ (a b c : ℝ),
  (∀ x y : ℝ, x^2 + 2*x + y^2 = 0 → (x + 1)^2 + y^2 = 1) ∧ 
  (a*1 + b*1 = 0) ∧
  (a*x + b*y + c = 0 ↔ x - y + 1 = 0) :=
by sorry

end perpendicular_line_equation_l3059_305940


namespace max_coach_handshakes_zero_l3059_305938

/-- The total number of handshakes in the tournament -/
def total_handshakes : ℕ := 465

/-- The number of players in the tournament -/
def num_players : ℕ := 31

/-- The number of handshakes between players -/
def player_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of handshakes the coach participated in -/
def coach_handshakes : ℕ := total_handshakes - player_handshakes num_players

theorem max_coach_handshakes_zero :
  coach_handshakes = 0 ∧ 
  ∀ n : ℕ, n > num_players → player_handshakes n > total_handshakes := by
  sorry


end max_coach_handshakes_zero_l3059_305938


namespace donation_problem_l3059_305926

theorem donation_problem (first_total second_total : ℕ) 
  (donor_ratio : ℚ) (avg_diff : ℕ) :
  first_total = 60000 →
  second_total = 150000 →
  donor_ratio = 3/2 →
  avg_diff = 20 →
  ∃ (first_donors : ℕ),
    first_donors = 2000 ∧
    (donor_ratio * first_donors : ℚ) = 3000 ∧
    (second_total : ℚ) / (donor_ratio * first_donors) - 
    (first_total : ℚ) / first_donors = avg_diff :=
by sorry

end donation_problem_l3059_305926


namespace root_difference_implies_k_value_l3059_305972

theorem root_difference_implies_k_value (k : ℝ) :
  (∃ r s : ℝ, r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0 ∧
   (r+3)^2 - k*(r+3) + 10 = 0 ∧ (s+3)^2 - k*(s+3) + 10 = 0) →
  k = 3 := by
sorry

end root_difference_implies_k_value_l3059_305972


namespace other_coin_denomination_l3059_305915

/-- Given a total of 336 coins with a total value of 7100 paise,
    where 260 of the coins are 20 paise coins,
    prove that the denomination of the other type of coin is 25 paise. -/
theorem other_coin_denomination
  (total_coins : ℕ)
  (total_value : ℕ)
  (twenty_paise_coins : ℕ)
  (h_total_coins : total_coins = 336)
  (h_total_value : total_value = 7100)
  (h_twenty_paise_coins : twenty_paise_coins = 260) :
  let other_coins := total_coins - twenty_paise_coins
  let other_denomination := (total_value - 20 * twenty_paise_coins) / other_coins
  other_denomination = 25 :=
by sorry

end other_coin_denomination_l3059_305915


namespace modified_cube_surface_area_l3059_305923

/-- Represents the structure of the cube after modifications -/
structure ModifiedCube where
  initial_size : Nat
  small_cube_size : Nat
  removed_center_cubes : Nat
  removed_per_small_cube : Nat

/-- Calculates the surface area of the modified cube structure -/
def surface_area (c : ModifiedCube) : Nat :=
  let remaining_small_cubes := c.initial_size^3 / c.small_cube_size^3 - c.removed_center_cubes
  let surface_per_small_cube := 6 * c.small_cube_size^2 + 12 -- Original surface + newly exposed
  remaining_small_cubes * surface_per_small_cube

/-- Theorem stating the surface area of the specific modified cube -/
theorem modified_cube_surface_area :
  let c : ModifiedCube := {
    initial_size := 12,
    small_cube_size := 3,
    removed_center_cubes := 7,
    removed_per_small_cube := 9
  }
  surface_area c = 3762 := by sorry

end modified_cube_surface_area_l3059_305923


namespace recipe_calculation_l3059_305906

/-- Represents the relationship between flour, cookies, and sugar -/
structure RecipeRelation where
  flour_to_cookies : ℝ → ℝ  -- Function from flour to cookies
  flour_to_sugar : ℝ → ℝ    -- Function from flour to sugar

/-- Given the recipe relationships, prove the number of cookies and amount of sugar for 4 cups of flour -/
theorem recipe_calculation (r : RecipeRelation) 
  (h1 : r.flour_to_cookies 3 = 24)  -- 24 cookies from 3 cups of flour
  (h2 : r.flour_to_sugar 3 = 1.5)   -- 1.5 cups of sugar for 3 cups of flour
  (h3 : ∀ x y, r.flour_to_cookies (x * y) = r.flour_to_cookies x * y)  -- Linear relationship for cookies
  (h4 : ∀ x y, r.flour_to_sugar (x * y) = r.flour_to_sugar x * y)      -- Linear relationship for sugar
  : r.flour_to_cookies 4 = 32 ∧ r.flour_to_sugar 4 = 2 := by
  sorry

#check recipe_calculation

end recipe_calculation_l3059_305906


namespace solution_of_equation_l3059_305921

theorem solution_of_equation (x : ℝ) : 2 * x - 4 * x = 0 ↔ x = 0 := by
  sorry

end solution_of_equation_l3059_305921


namespace smallest_integer_with_given_remainders_l3059_305941

theorem smallest_integer_with_given_remainders : ∃ x : ℕ, 
  (x > 0) ∧ 
  (x % 3 = 2) ∧ 
  (x % 4 = 3) ∧ 
  (x % 5 = 4) ∧ 
  (∀ y : ℕ, y > 0 → y % 3 = 2 → y % 4 = 3 → y % 5 = 4 → x ≤ y) ∧
  (x = 59) := by
sorry

end smallest_integer_with_given_remainders_l3059_305941


namespace principal_determination_l3059_305988

/-- Given a principal amount and an unknown interest rate, if increasing the
    interest rate by 6 percentage points results in Rs. 30 more interest over 1 year,
    then the principal must be Rs. 500. -/
theorem principal_determination (P R : ℝ) (h : P * (R + 6) / 100 - P * R / 100 = 30) :
  P = 500 := by
  sorry

end principal_determination_l3059_305988


namespace triangle_construction_theorem_l3059_305950

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Midpoint (A B M : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def OnLine (P : Point) (L : Line) : Prop :=
  L.a * P.x + L.b * P.y + L.c = 0

def AngleBisector (A B C : Point) (L : Line) : Prop :=
  -- This is a simplified definition and may need to be expanded
  OnLine A L ∧ OnLine B L

-- The main theorem
theorem triangle_construction_theorem :
  ∀ (N M : Point) (l : Line),
  ∃ (A B C : Point),
    Midpoint A C N ∧
    Midpoint B C M ∧
    AngleBisector A B C l :=
sorry

end triangle_construction_theorem_l3059_305950
