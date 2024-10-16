import Mathlib

namespace NUMINAMATH_CALUDE_f_monotone_increasing_f_monotone_decreasing_f_not_always_above_a_l1139_113900

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

-- Theorem 1: f(x) is monotonically increasing on ℝ iff a ≤ 0
theorem f_monotone_increasing (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 0 :=
sorry

-- Theorem 2: f(x) is monotonically decreasing on (-1, 1) iff a ≥ 3
theorem f_monotone_decreasing (a : ℝ) :
  (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f a x > f a y) ↔ a ≥ 3 :=
sorry

-- Theorem 3: ∃x ∈ ℝ, f(x) < a
theorem f_not_always_above_a (a : ℝ) :
  ∃ x : ℝ, f a x < a :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_f_monotone_decreasing_f_not_always_above_a_l1139_113900


namespace NUMINAMATH_CALUDE_erica_pie_percentage_l1139_113927

theorem erica_pie_percentage :
  ∀ (apple_fraction cherry_fraction : ℚ),
    apple_fraction = 1/5 →
    cherry_fraction = 3/4 →
    (apple_fraction + cherry_fraction) * 100 = 95 := by
  sorry

end NUMINAMATH_CALUDE_erica_pie_percentage_l1139_113927


namespace NUMINAMATH_CALUDE_unique_right_triangle_completion_l1139_113953

/-- A function that checks if three side lengths form a right triangle -/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that there is exactly one integer side length 
    that can complete a right triangle with sides 8 and 15 -/
theorem unique_right_triangle_completion :
  ∃! x : ℕ, is_right_triangle 8 15 x :=
sorry

end NUMINAMATH_CALUDE_unique_right_triangle_completion_l1139_113953


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1139_113963

theorem inequality_equivalence (x : ℝ) : 
  (x + 3) / 2 - (5 * x - 1) / 5 ≥ 0 ↔ x ≤ 17 / 5 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1139_113963


namespace NUMINAMATH_CALUDE_calculator_squaring_min_presses_1000_eq_3_l1139_113943

def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | m + 1 => (repeated_square x m) ^ 2

theorem calculator_squaring (target : ℕ) : 
  ∃ (n : ℕ), repeated_square 3 n > target ∧ 
  ∀ (m : ℕ), m < n → repeated_square 3 m ≤ target := by
  sorry

def min_presses (target : ℕ) : ℕ :=
  Nat.find (calculator_squaring target)

theorem min_presses_1000_eq_3 : min_presses 1000 = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculator_squaring_min_presses_1000_eq_3_l1139_113943


namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_has_odd_number_of_factors_196_is_less_than_200_largest_number_with_odd_factors_l1139_113930

def has_odd_number_of_factors (n : ℕ) : Prop :=
  Odd (Nat.card {d : ℕ | d ∣ n ∧ d > 0})

theorem greatest_number_with_odd_factors :
  ∀ n : ℕ, n < 200 → has_odd_number_of_factors n → n ≤ 196 :=
by sorry

theorem has_odd_number_of_factors_196 : has_odd_number_of_factors 196 :=
by sorry

theorem is_less_than_200 : 196 < 200 :=
by sorry

theorem largest_number_with_odd_factors :
  ∃ n : ℕ, n < 200 ∧ has_odd_number_of_factors n ∧
  ∀ m : ℕ, m < 200 → has_odd_number_of_factors m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_has_odd_number_of_factors_196_is_less_than_200_largest_number_with_odd_factors_l1139_113930


namespace NUMINAMATH_CALUDE_preimages_of_one_l1139_113974

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem preimages_of_one (x : ℝ) : 
  f x = 1 ↔ x = -1 ∨ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_preimages_of_one_l1139_113974


namespace NUMINAMATH_CALUDE_total_cards_l1139_113965

theorem total_cards (initial_cards : ℕ) (added_cards : ℕ) : 
  initial_cards = 4 → added_cards = 3 → initial_cards + added_cards = 7 :=
by sorry

end NUMINAMATH_CALUDE_total_cards_l1139_113965


namespace NUMINAMATH_CALUDE_square_of_product_l1139_113914

theorem square_of_product (a b : ℝ) : (-2 * a * b^3)^2 = 4 * a^2 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_l1139_113914


namespace NUMINAMATH_CALUDE_rhinestones_needed_proof_l1139_113977

/-- Given a total number of rhinestones needed, calculate the number still needed
    after buying one-third and finding one-fifth of the total. -/
def rhinestones_still_needed (total : ℕ) : ℕ :=
  total - (total / 3) - (total / 5)

/-- Theorem stating that for 45 rhinestones, the number still needed is 21. -/
theorem rhinestones_needed_proof :
  rhinestones_still_needed 45 = 21 := by
  sorry

#eval rhinestones_still_needed 45

end NUMINAMATH_CALUDE_rhinestones_needed_proof_l1139_113977


namespace NUMINAMATH_CALUDE_triangle_side_not_eight_l1139_113910

/-- A triangle with side lengths a, b, and c exists if and only if the sum of any two sides is greater than the third side for all combinations. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: In a triangle with side lengths 3, 5, and x, x cannot be 8. -/
theorem triangle_side_not_eight :
  ¬ (triangle_inequality 3 5 8) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_not_eight_l1139_113910


namespace NUMINAMATH_CALUDE_button_probability_l1139_113921

/-- Represents a jar containing buttons of two colors -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- Represents the state of two jars after button transfer -/
structure JarState where
  jarA : Jar
  jarB : Jar

def initialJarA : Jar := { red := 7, blue := 9 }

def buttonTransfer (initial : Jar) : JarState :=
  { jarA := { red := initial.red - 3, blue := initial.blue - 2 },
    jarB := { red := 3, blue := 2 } }

def probability_red (jar : Jar) : ℚ :=
  jar.red / (jar.red + jar.blue)

theorem button_probability (initial : Jar := initialJarA) :
  let final := buttonTransfer initial
  let probA := probability_red final.jarA
  let probB := probability_red final.jarB
  probA * probB = 12 / 55 := by
  sorry

end NUMINAMATH_CALUDE_button_probability_l1139_113921


namespace NUMINAMATH_CALUDE_min_sum_for_product_3006_l1139_113994

theorem min_sum_for_product_3006 (a b c : ℕ+) (h : a * b * c = 3006) :
  (∀ x y z : ℕ+, x * y * z = 3006 → a + b + c ≤ x + y + z) ∧ a + b + c = 105 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_for_product_3006_l1139_113994


namespace NUMINAMATH_CALUDE_quadratic_properties_l1139_113949

-- Define the quadratic function
def quadratic (b c x : ℝ) : ℝ := -x^2 + b*x + c

theorem quadratic_properties :
  ∀ (b c : ℝ),
  -- Part 1
  (quadratic b c (-1) = 0 ∧ quadratic b c 3 = 0 →
    ∃ x, ∀ y, quadratic b c y ≤ quadratic b c x ∧ quadratic b c x = 4) ∧
  -- Part 2
  (c = -5 ∧ (∃! x, quadratic b c x = 1) →
    b = 2 * Real.sqrt 6 ∨ b = -2 * Real.sqrt 6) ∧
  -- Part 3
  (c = b^2 ∧ (∃ x, b ≤ x ∧ x ≤ b + 3 ∧
    ∀ y, b ≤ y ∧ y ≤ b + 3 → quadratic b c y ≤ quadratic b c x) ∧
    quadratic b c x = 20 →
    b = 2 * Real.sqrt 5 ∨ b = -4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1139_113949


namespace NUMINAMATH_CALUDE_fraction_numerator_l1139_113906

theorem fraction_numerator (y : ℝ) (h : y > 0) :
  ∃ x : ℝ, (x / y) * y + (3 * y) / 10 = 0.7 * y ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_l1139_113906


namespace NUMINAMATH_CALUDE_investment_problem_l1139_113918

/-- Given two investors P and Q, where P invested 40000 and their profit ratio is 2:3,
    prove that Q's investment is 60000. -/
theorem investment_problem (P Q : ℕ) (h1 : P = 40000) (h2 : 2 * Q = 3 * P) : Q = 60000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l1139_113918


namespace NUMINAMATH_CALUDE_remaining_black_portion_l1139_113942

/-- The fraction of black area remaining after one transformation -/
def black_fraction : ℚ := 3 / 4

/-- The number of transformations applied -/
def num_transformations : ℕ := 5

/-- The theorem stating the remaining black portion after transformations -/
theorem remaining_black_portion :
  black_fraction ^ num_transformations = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_remaining_black_portion_l1139_113942


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1139_113957

theorem sum_of_three_numbers : ∀ (n₁ n₂ n₃ : ℕ),
  n₂ = 72 →
  n₁ = 2 * n₂ →
  n₃ = n₁ / 3 →
  n₁ + n₂ + n₃ = 264 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1139_113957


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l1139_113986

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l1139_113986


namespace NUMINAMATH_CALUDE_sam_bought_nine_cans_l1139_113960

/-- The number of coupons Sam had -/
def num_coupons : ℕ := 5

/-- The discount per coupon in cents -/
def discount_per_coupon : ℕ := 25

/-- The amount Sam paid in cents -/
def amount_paid : ℕ := 2000

/-- The change Sam received in cents -/
def change_received : ℕ := 550

/-- The cost of each can of tuna in cents -/
def cost_per_can : ℕ := 175

/-- The number of cans Sam bought -/
def num_cans : ℕ := (amount_paid - change_received + num_coupons * discount_per_coupon) / cost_per_can

theorem sam_bought_nine_cans : num_cans = 9 := by
  sorry

end NUMINAMATH_CALUDE_sam_bought_nine_cans_l1139_113960


namespace NUMINAMATH_CALUDE_rectangle_formations_6_7_l1139_113905

/-- The number of ways to choose 2 items from n items -/
def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of ways to form a rectangle given h horizontal lines and v vertical lines -/
def rectangle_formations (h v : ℕ) : ℕ := choose_2 h * choose_2 v

/-- Theorem stating that with 6 horizontal and 7 vertical lines, there are 315 ways to form a rectangle -/
theorem rectangle_formations_6_7 : rectangle_formations 6 7 = 315 := by sorry

end NUMINAMATH_CALUDE_rectangle_formations_6_7_l1139_113905


namespace NUMINAMATH_CALUDE_min_difference_triangle_sides_l1139_113984

theorem min_difference_triangle_sides (a b c : ℕ) : 
  a + b + c = 2007 →
  a < b →
  b ≤ c →
  (∀ a' b' c' : ℕ, a' + b' + c' = 2007 → a' < b' → b' ≤ c' → b - a ≤ b' - a') →
  b - a = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_difference_triangle_sides_l1139_113984


namespace NUMINAMATH_CALUDE_chris_teslas_l1139_113940

theorem chris_teslas (elon sam chris : ℕ) : 
  elon = 13 →
  elon = sam + 10 →
  sam * 2 = chris →
  chris = 6 := by sorry

end NUMINAMATH_CALUDE_chris_teslas_l1139_113940


namespace NUMINAMATH_CALUDE_journey_length_l1139_113916

theorem journey_length :
  ∀ (total : ℚ),
  (1 / 4 : ℚ) * total + 24 + (1 / 6 : ℚ) * total = total →
  total = 288 / 7 := by
sorry

end NUMINAMATH_CALUDE_journey_length_l1139_113916


namespace NUMINAMATH_CALUDE_not_divisible_by_121_l1139_113937

theorem not_divisible_by_121 (n : ℤ) : ¬(∃ (k : ℤ), n^2 + 3*n + 5 = 121*k ∨ n^2 - 3*n + 5 = 121*k) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_121_l1139_113937


namespace NUMINAMATH_CALUDE_no_p_q_for_all_x_divisible_by_3_l1139_113946

theorem no_p_q_for_all_x_divisible_by_3 : 
  ¬ ∃ (p q : ℤ), ∀ (x : ℤ), (3 : ℤ) ∣ (x^2 + p*x + q) := by
  sorry

end NUMINAMATH_CALUDE_no_p_q_for_all_x_divisible_by_3_l1139_113946


namespace NUMINAMATH_CALUDE_probability_two_girls_l1139_113958

theorem probability_two_girls (total : Nat) (girls : Nat) (selected : Nat) : 
  total = 6 → girls = 4 → selected = 2 →
  (Nat.choose girls selected : Rat) / (Nat.choose total selected : Rat) = 2/5 := by
sorry

end NUMINAMATH_CALUDE_probability_two_girls_l1139_113958


namespace NUMINAMATH_CALUDE_type_b_soda_cans_l1139_113924

/-- The number of cans of type B soda that can be purchased for a given amount of money -/
theorem type_b_soda_cans 
  (T : ℕ) -- number of type A cans
  (P : ℕ) -- price in quarters for T cans of type A
  (R : ℚ) -- amount of dollars available
  (h1 : P > 0) -- ensure division by P is valid
  (h2 : T > 0) -- ensure division by T is valid
  : (2 * R * T.cast) / P.cast = (4 * R * T.cast) / (2 * P.cast) := by
  sorry

end NUMINAMATH_CALUDE_type_b_soda_cans_l1139_113924


namespace NUMINAMATH_CALUDE_marbles_lost_l1139_113995

theorem marbles_lost (initial : ℕ) (current : ℕ) (lost : ℕ) 
  (h1 : initial = 19) 
  (h2 : current = 8) 
  (h3 : lost = initial - current) : lost = 11 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_l1139_113995


namespace NUMINAMATH_CALUDE_solution_to_system_l1139_113982

theorem solution_to_system (x y z : ℝ) 
  (eq1 : x = 1 + Real.sqrt (y - z^2))
  (eq2 : y = 1 + Real.sqrt (z - x^2))
  (eq3 : z = 1 + Real.sqrt (x - y^2)) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_to_system_l1139_113982


namespace NUMINAMATH_CALUDE_midpoint_ratio_range_l1139_113909

/-- Given two points P and Q on different lines, with midpoint M satisfying certain conditions,
    prove that the ratio of y₀ to x₀ (coordinates of M) is between -1 and -1/3 -/
theorem midpoint_ratio_range (P Q M : ℝ × ℝ) (x₀ y₀ : ℝ) :
  (P.1 + P.2 = 1) →  -- P lies on x + y = 1
  (Q.1 + Q.2 = -3) →  -- Q lies on x + y = -3
  (M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) →  -- M is midpoint of PQ
  (M = (x₀, y₀)) →  -- M has coordinates (x₀, y₀)
  (x₀ - y₀ + 2 < 0) →  -- given condition
  (-1 < y₀ / x₀ ∧ y₀ / x₀ < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_ratio_range_l1139_113909


namespace NUMINAMATH_CALUDE_sin_tan_greater_than_square_l1139_113978

theorem sin_tan_greater_than_square (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) : 
  Real.sin x * Real.tan x > x^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_tan_greater_than_square_l1139_113978


namespace NUMINAMATH_CALUDE_vanessa_music_files_l1139_113971

/-- The number of music files Vanessa initially had -/
def initial_music_files : ℕ := 13

/-- The number of video files Vanessa initially had -/
def video_files : ℕ := 30

/-- The number of files deleted -/
def deleted_files : ℕ := 10

/-- The number of files remaining after deletion -/
def remaining_files : ℕ := 33

theorem vanessa_music_files :
  initial_music_files + video_files = remaining_files + deleted_files :=
by sorry

end NUMINAMATH_CALUDE_vanessa_music_files_l1139_113971


namespace NUMINAMATH_CALUDE_stream_speed_l1139_113955

/-- Proves that the speed of a stream is 4 km/hr given the conditions of the boat's travel. -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 13 →
  downstream_distance = 68 →
  downstream_time = 4 →
  (boat_speed + (downstream_distance / downstream_time - boat_speed)) = 17 := by
  sorry

#check stream_speed

end NUMINAMATH_CALUDE_stream_speed_l1139_113955


namespace NUMINAMATH_CALUDE_chord_length_intercepted_by_line_l1139_113915

/-- The chord length intercepted by a line on a circle -/
theorem chord_length_intercepted_by_line (x y : ℝ) : 
  let circle : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 4}
  let line : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}
  let chord_length := Real.sqrt (8 : ℝ)
  (∃ p q : ℝ × ℝ, p ∈ circle ∧ q ∈ circle ∧ p ∈ line ∧ q ∈ line ∧ 
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = chord_length^2) :=
by
  sorry


end NUMINAMATH_CALUDE_chord_length_intercepted_by_line_l1139_113915


namespace NUMINAMATH_CALUDE_oliver_monster_club_cards_l1139_113969

/-- Represents Oliver's card collection --/
structure CardCollection where
  alien_baseball : ℕ
  monster_club : ℕ
  battle_gremlins : ℕ

/-- The conditions of Oliver's card collection --/
def oliver_collection : CardCollection :=
  { alien_baseball := 18,
    monster_club := 27,
    battle_gremlins := 72 }

/-- Theorem stating the number of Monster Club cards Oliver has --/
theorem oliver_monster_club_cards :
  oliver_collection.monster_club = 27 ∧
  oliver_collection.monster_club = (3 / 2 : ℚ) * oliver_collection.alien_baseball ∧
  oliver_collection.battle_gremlins = 72 ∧
  oliver_collection.battle_gremlins = 4 * oliver_collection.alien_baseball :=
by
  sorry

end NUMINAMATH_CALUDE_oliver_monster_club_cards_l1139_113969


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l1139_113944

theorem chess_tournament_participants (n : ℕ) : 
  (∃ (y : ℚ), 2 * y + n * y = (n + 2) * (n + 1) / 2) → 
  (n = 7 ∨ n = 14) := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l1139_113944


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1139_113981

def P : Set ℝ := {-2, 0, 2, 4}
def Q : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_P_and_Q : P ∩ Q = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1139_113981


namespace NUMINAMATH_CALUDE_largest_table_sum_l1139_113938

def numbers : List ℕ := [2, 3, 5, 7, 11, 17, 19]

def is_valid_arrangement (top : List ℕ) (left : List ℕ) : Prop :=
  top.length = 3 ∧ left.length = 3 ∧ (top ++ left).toFinset ⊆ numbers.toFinset

def table_sum (top : List ℕ) (left : List ℕ) : ℕ :=
  (top.sum * left.sum)

theorem largest_table_sum :
  ∀ (top left : List ℕ), is_valid_arrangement top left →
  table_sum top left ≤ 1024 :=
sorry

end NUMINAMATH_CALUDE_largest_table_sum_l1139_113938


namespace NUMINAMATH_CALUDE_lines_parallel_or_skew_l1139_113962

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the subset relation for lines and planes
variable (line_in_plane : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (line_parallel : Line → Line → Prop)

-- Define the skew relation for lines
variable (line_skew : Line → Line → Prop)

-- Theorem statement
theorem lines_parallel_or_skew
  (α β : Plane) (a b : Line)
  (h_parallel : plane_parallel α β)
  (h_a_in_α : line_in_plane a α)
  (h_b_in_β : line_in_plane b β) :
  line_parallel a b ∨ line_skew a b :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_or_skew_l1139_113962


namespace NUMINAMATH_CALUDE_min_distance_line_circle_l1139_113945

/-- The minimum distance between a point on the given line and a point on the given circle is √5/5 -/
theorem min_distance_line_circle :
  let line := {p : ℝ × ℝ | ∃ t : ℝ, p.1 = t ∧ p.2 = 6 - 2*t}
  let circle := {q : ℝ × ℝ | (q.1 - 1)^2 + (q.2 + 2)^2 = 5}
  ∃ d : ℝ, d = Real.sqrt 5 / 5 ∧
    ∀ p ∈ line, ∀ q ∈ circle,
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d ∧
      ∃ p' ∈ line, ∃ q' ∈ circle,
        Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) = d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_circle_l1139_113945


namespace NUMINAMATH_CALUDE_tree_increase_factor_l1139_113912

theorem tree_increase_factor (initial_maples : ℝ) (initial_lindens : ℝ) 
  (spring_total : ℝ) (autumn_total : ℝ) : 
  initial_maples / (initial_maples + initial_lindens) = 3/5 →
  initial_maples / spring_total = 1/5 →
  initial_maples / autumn_total = 3/5 →
  autumn_total / (initial_maples + initial_lindens) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_increase_factor_l1139_113912


namespace NUMINAMATH_CALUDE_rectangle_formation_count_l1139_113931

theorem rectangle_formation_count (h : ℕ) (v : ℕ) : h = 6 → v = 5 → Nat.choose h 2 * Nat.choose v 2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formation_count_l1139_113931


namespace NUMINAMATH_CALUDE_p_true_and_q_false_l1139_113936

-- Define proposition p
def p : Prop := ∀ z : ℂ, (z - Complex.I) * (-Complex.I) = 5 → z = 6 * Complex.I

-- Define proposition q
def q : Prop := Complex.im ((1 + Complex.I) / (1 + 2 * Complex.I)) = -1/5

-- Theorem to prove
theorem p_true_and_q_false : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_p_true_and_q_false_l1139_113936


namespace NUMINAMATH_CALUDE_concyclicity_equivalence_l1139_113987

-- Define the points
variable (A B C D P E F G H O₁ O₂ O₃ O₄ : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect_at (A B C D P : EuclideanPlane) : Prop := sorry

-- Define midpoints
def is_midpoint (M A B : EuclideanPlane) : Prop := sorry

-- Define circumcenter
def is_circumcenter (O P Q R : EuclideanPlane) : Prop := sorry

-- Define concyclicity
def are_concyclic (P Q R S : EuclideanPlane) : Prop := sorry

-- Main theorem
theorem concyclicity_equivalence 
  (h_quad : is_convex_quadrilateral A B C D)
  (h_diag : diagonals_intersect_at A B C D P)
  (h_mid_E : is_midpoint E A B)
  (h_mid_F : is_midpoint F B C)
  (h_mid_G : is_midpoint G C D)
  (h_mid_H : is_midpoint H D A)
  (h_circ_O₁ : is_circumcenter O₁ P H E)
  (h_circ_O₂ : is_circumcenter O₂ P E F)
  (h_circ_O₃ : is_circumcenter O₃ P F G)
  (h_circ_O₄ : is_circumcenter O₄ P G H) :
  are_concyclic O₁ O₂ O₃ O₄ ↔ are_concyclic A B C D := by sorry

end NUMINAMATH_CALUDE_concyclicity_equivalence_l1139_113987


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1139_113973

/-- The standard equation of a hyperbola given its asymptotes and shared foci with an ellipse -/
theorem hyperbola_standard_equation
  (asymptote_slope : ℝ)
  (ellipse_a : ℝ)
  (ellipse_b : ℝ)
  (h_asymptote : asymptote_slope = 2)
  (h_ellipse : ellipse_a^2 = 49 ∧ ellipse_b^2 = 24) :
  ∃ (a b : ℝ), a^2 = 25 ∧ b^2 = 100 ∧
    ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1139_113973


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1139_113983

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 5 * x - 3
  ∃ x₁ x₂ : ℝ, x₁ = -3 ∧ x₂ = 1/2 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1139_113983


namespace NUMINAMATH_CALUDE_weight_replacement_l1139_113956

theorem weight_replacement (initial_count : ℕ) (average_increase : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  average_increase = 2.5 →
  new_weight = 80 →
  ∃ (old_weight : ℝ),
    old_weight = new_weight - (initial_count * average_increase) ∧
    old_weight = 60 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l1139_113956


namespace NUMINAMATH_CALUDE_sara_pumpkins_left_l1139_113923

def pumpkins_left (initial : ℕ) (eaten_by_rabbits : ℕ) (eaten_by_raccoons : ℕ) (given_away : ℕ) : ℕ :=
  initial - eaten_by_rabbits - eaten_by_raccoons - given_away

theorem sara_pumpkins_left : 
  pumpkins_left 43 23 5 7 = 8 := by sorry

end NUMINAMATH_CALUDE_sara_pumpkins_left_l1139_113923


namespace NUMINAMATH_CALUDE_distance_origin_to_line_l1139_113926

/-- The distance from the origin to the line x + √3y - 2 = 0 is 1 -/
theorem distance_origin_to_line : 
  let line := {(x, y) : ℝ × ℝ | x + Real.sqrt 3 * y - 2 = 0}
  ∃ d : ℝ, d = 1 ∧ ∀ (p : ℝ × ℝ), p ∈ line → Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_distance_origin_to_line_l1139_113926


namespace NUMINAMATH_CALUDE_apple_collection_l1139_113935

theorem apple_collection (A K : ℕ) (hA : A > 0) (hK : K > 0) : 
  let T := A + K
  (A = (K * 100) / T) → (K = (A * 100) / T) → (A = 50 ∧ K = 50) :=
by sorry

end NUMINAMATH_CALUDE_apple_collection_l1139_113935


namespace NUMINAMATH_CALUDE_patel_family_concert_cost_l1139_113993

theorem patel_family_concert_cost : 
  let regular_ticket_price : ℚ := 7.50 / (1 - 0.20)
  let children_ticket_price : ℚ := regular_ticket_price * (1 - 0.60)
  let senior_ticket_price : ℚ := 7.50
  let num_tickets_per_generation : ℕ := 2
  let handling_fee : ℚ := 5

  (num_tickets_per_generation * senior_ticket_price + 
   num_tickets_per_generation * regular_ticket_price + 
   num_tickets_per_generation * children_ticket_price + 
   handling_fee) = 46.25 := by
sorry


end NUMINAMATH_CALUDE_patel_family_concert_cost_l1139_113993


namespace NUMINAMATH_CALUDE_race_finish_times_l1139_113947

/-- Race parameters and results -/
structure RaceData where
  malcolm_speed : ℝ  -- Malcolm's speed in minutes per mile
  joshua_speed : ℝ   -- Joshua's speed in minutes per mile
  ellie_speed : ℝ    -- Ellie's speed in minutes per mile
  race_distance : ℝ  -- Race distance in miles

def finish_time (speed : ℝ) (distance : ℝ) : ℝ := speed * distance

/-- Theorem stating the time differences for Joshua and Ellie compared to Malcolm -/
theorem race_finish_times (data : RaceData) 
  (h_malcolm : data.malcolm_speed = 5)
  (h_joshua : data.joshua_speed = 7)
  (h_ellie : data.ellie_speed = 6)
  (h_distance : data.race_distance = 15) :
  let malcolm_time := finish_time data.malcolm_speed data.race_distance
  let joshua_time := finish_time data.joshua_speed data.race_distance
  let ellie_time := finish_time data.ellie_speed data.race_distance
  (joshua_time - malcolm_time = 30 ∧ ellie_time - malcolm_time = 15) := by
  sorry


end NUMINAMATH_CALUDE_race_finish_times_l1139_113947


namespace NUMINAMATH_CALUDE_shelly_has_enough_thread_l1139_113929

/-- Represents the keychain making scenario for Shelly's friends --/
structure KeychainScenario where
  class_friends : Nat
  club_friends : Nat
  sports_friends : Nat
  class_thread : Nat
  club_thread : Nat
  sports_thread : Nat
  available_thread : Nat

/-- Calculates the total thread needed and checks if it's sufficient --/
def thread_calculation (scenario : KeychainScenario) : 
  (Bool × Nat) :=
  let total_needed := 
    scenario.class_friends * scenario.class_thread +
    scenario.club_friends * scenario.club_thread +
    scenario.sports_friends * scenario.sports_thread
  let is_sufficient := total_needed ≤ scenario.available_thread
  let remaining := scenario.available_thread - total_needed
  (is_sufficient, remaining)

/-- Theorem stating that Shelly has enough thread and calculates the remaining amount --/
theorem shelly_has_enough_thread (scenario : KeychainScenario) 
  (h1 : scenario.class_friends = 10)
  (h2 : scenario.club_friends = 20)
  (h3 : scenario.sports_friends = 5)
  (h4 : scenario.class_thread = 18)
  (h5 : scenario.club_thread = 24)
  (h6 : scenario.sports_thread = 30)
  (h7 : scenario.available_thread = 1200) :
  thread_calculation scenario = (true, 390) := by
  sorry

end NUMINAMATH_CALUDE_shelly_has_enough_thread_l1139_113929


namespace NUMINAMATH_CALUDE_complex_fraction_division_l1139_113988

theorem complex_fraction_division : 
  (5 / (8 / 13)) / (10 / 7) = 91 / 16 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_division_l1139_113988


namespace NUMINAMATH_CALUDE_sum_of_squares_l1139_113904

theorem sum_of_squares (x y z : ℝ) 
  (h_arithmetic : (x + y + z) / 3 = 9)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 405 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1139_113904


namespace NUMINAMATH_CALUDE_binomial_60_3_l1139_113998

theorem binomial_60_3 : Nat.choose 60 3 = 57020 := by sorry

end NUMINAMATH_CALUDE_binomial_60_3_l1139_113998


namespace NUMINAMATH_CALUDE_log_four_eighteen_l1139_113950

theorem log_four_eighteen (a b : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 3 / Real.log 10 = b) :
  Real.log 18 / Real.log 4 = (a + 2*b) / (2*a) := by sorry

end NUMINAMATH_CALUDE_log_four_eighteen_l1139_113950


namespace NUMINAMATH_CALUDE_equation_system_solution_l1139_113980

theorem equation_system_solution : ∃! (a b c d e f : ℕ),
  (a ∈ Finset.range 10) ∧
  (b ∈ Finset.range 10) ∧
  (c ∈ Finset.range 10) ∧
  (d ∈ Finset.range 10) ∧
  (e ∈ Finset.range 10) ∧
  (f ∈ Finset.range 10) ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧
  (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧
  (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧
  (d ≠ e) ∧ (d ≠ f) ∧
  (e ≠ f) ∧
  (20 * (a - 8) = 20) ∧
  (b / 2 + 17 = 20) ∧
  (c * 8 - 4 = 20) ∧
  ((d + 8) / 12 = 1) ∧
  (4 * e = 20) ∧
  (20 * (f - 2) = 100) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_system_solution_l1139_113980


namespace NUMINAMATH_CALUDE_deal_or_no_deal_boxes_l1139_113920

theorem deal_or_no_deal_boxes (total_boxes : ℕ) (high_value_boxes : ℕ) (eliminated_boxes : ℕ) : 
  total_boxes = 30 →
  high_value_boxes = 7 →
  (high_value_boxes : ℚ) / ((total_boxes - eliminated_boxes) : ℚ) ≥ 2 / 3 →
  eliminated_boxes ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_boxes_l1139_113920


namespace NUMINAMATH_CALUDE_volleyball_starters_count_l1139_113990

def volleyball_team_size : ℕ := 14
def triplet_size : ℕ := 3
def starter_size : ℕ := 6

def choose_starters (team_size triplet_size starter_size : ℕ) : ℕ :=
  let non_triplet_size := team_size - triplet_size
  let remaining_spots := starter_size - 2
  triplet_size * Nat.choose non_triplet_size remaining_spots

theorem volleyball_starters_count :
  choose_starters volleyball_team_size triplet_size starter_size = 990 :=
sorry

end NUMINAMATH_CALUDE_volleyball_starters_count_l1139_113990


namespace NUMINAMATH_CALUDE_equation_solutions_l1139_113934

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => x * (x - 3) - 10
  (f 5 = 0 ∧ f (-2) = 0) ∧ ∀ x : ℝ, f x = 0 → (x = 5 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1139_113934


namespace NUMINAMATH_CALUDE_solution_values_l1139_113975

theorem solution_values (a : ℝ) : (a - 2) ^ (a + 1) = 1 → a = -1 ∨ a = 3 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_values_l1139_113975


namespace NUMINAMATH_CALUDE_renovation_calculation_l1139_113922

/-- Represents the dimensions and characteristics of a bedroom --/
structure Bedroom where
  length : ℝ
  width : ℝ
  height : ℝ
  unpaintable_area : ℝ
  fixed_furniture_area : ℝ

/-- Calculates the total area to be painted in all bedrooms --/
def total_paintable_area (b : Bedroom) (num_bedrooms : ℕ) : ℝ :=
  num_bedrooms * (2 * (b.length * b.height + b.width * b.height) - b.unpaintable_area)

/-- Calculates the total carpet area for all bedrooms --/
def total_carpet_area (b : Bedroom) (num_bedrooms : ℕ) : ℝ :=
  num_bedrooms * (b.length * b.width - b.fixed_furniture_area)

/-- Theorem stating the correct paintable area and carpet area --/
theorem renovation_calculation (b : Bedroom) (h1 : b.length = 14)
    (h2 : b.width = 11) (h3 : b.height = 9) (h4 : b.unpaintable_area = 70)
    (h5 : b.fixed_furniture_area = 24) :
    total_paintable_area b 4 = 1520 ∧ total_carpet_area b 4 = 520 := by
  sorry


end NUMINAMATH_CALUDE_renovation_calculation_l1139_113922


namespace NUMINAMATH_CALUDE_largest_B_divisible_by_three_l1139_113979

def seven_digit_number (B : ℕ) : ℕ := 4000000 + B * 100000 + 68251

theorem largest_B_divisible_by_three :
  ∀ B : ℕ, B ≤ 9 →
    (seven_digit_number B % 3 = 0) →
    B ≤ 7 ∧
    seven_digit_number 7 % 3 = 0 ∧
    (∀ C : ℕ, C > 7 → C ≤ 9 → seven_digit_number C % 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_B_divisible_by_three_l1139_113979


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_values_l1139_113932

theorem infinite_solutions_imply_values (a b : ℚ) : 
  (∀ x : ℚ, a * (2 * x + b) = 12 * x + 5) → 
  (a = 6 ∧ b = 5/6) := by
sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_values_l1139_113932


namespace NUMINAMATH_CALUDE_circle_angle_problem_l1139_113952

theorem circle_angle_problem (x y : ℝ) : 
  3 * x + 2 * y + 5 * x + 7 * x = 360 →
  x = y →
  x = 360 / 17 ∧ y = 360 / 17 := by
sorry

end NUMINAMATH_CALUDE_circle_angle_problem_l1139_113952


namespace NUMINAMATH_CALUDE_expression_evaluation_l1139_113959

theorem expression_evaluation : 12 - 7 + 11 * 4 + 8 - 10 * 2 + 6 / 2 - 3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1139_113959


namespace NUMINAMATH_CALUDE_solutions_to_equation_l1139_113997

theorem solutions_to_equation : 
  {(m, n) : ℕ × ℕ | 7^m - 3 * 2^n = 1} = {(1, 1), (2, 4)} := by sorry

end NUMINAMATH_CALUDE_solutions_to_equation_l1139_113997


namespace NUMINAMATH_CALUDE_sector_max_area_l1139_113992

theorem sector_max_area (r l : ℝ) (h_perimeter : 2 * r + l = 10) (h_positive : r > 0 ∧ l > 0) :
  (1 / 2) * l * r ≤ 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l1139_113992


namespace NUMINAMATH_CALUDE_sin_cos_tan_product_l1139_113964

theorem sin_cos_tan_product : 
  Real.sin (4/3 * Real.pi) * Real.cos (5/6 * Real.pi) * Real.tan (-4/3 * Real.pi) = -3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_tan_product_l1139_113964


namespace NUMINAMATH_CALUDE_sum_y_four_times_equals_four_y_l1139_113901

theorem sum_y_four_times_equals_four_y (y : ℝ) : y + y + y + y = 4 * y := by
  sorry

end NUMINAMATH_CALUDE_sum_y_four_times_equals_four_y_l1139_113901


namespace NUMINAMATH_CALUDE_june_production_l1139_113907

/-- Represents a restaurant's daily pizza and hot dog production. -/
structure RestaurantProduction where
  hotDogs : ℕ
  pizzaDifference : ℕ

/-- Calculates the total number of pizzas and hot dogs made in June. -/
def totalInJune (r : RestaurantProduction) : ℕ :=
  30 * (r.hotDogs + (r.hotDogs + r.pizzaDifference))

/-- Theorem stating the total production in June for a specific restaurant. -/
theorem june_production (r : RestaurantProduction) 
  (h1 : r.hotDogs = 60) 
  (h2 : r.pizzaDifference = 40) : 
  totalInJune r = 4800 := by
  sorry

#eval totalInJune ⟨60, 40⟩

end NUMINAMATH_CALUDE_june_production_l1139_113907


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1139_113991

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x - y = 6) : y = 2 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1139_113991


namespace NUMINAMATH_CALUDE_pyramid_solution_l1139_113911

structure NumberPyramid where
  row1_left : ℕ
  row1_right : ℕ
  row2_left : ℕ
  row2_right : ℕ
  row3_left : ℕ
  row3_middle : ℕ
  row3_right : ℕ

def is_valid_pyramid (p : NumberPyramid) : Prop :=
  p.row2_left = p.row1_left + p.row1_right ∧
  p.row2_right = p.row1_right + 660 ∧
  p.row3_left = p.row2_left * p.row1_left ∧
  p.row3_middle = p.row2_left * p.row2_right ∧
  p.row3_right = p.row2_right * 660

theorem pyramid_solution :
  ∃ (p : NumberPyramid), is_valid_pyramid p ∧ 
    p.row3_left = 28 ∧ p.row3_right = 630 ∧ p.row2_left = 13 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_solution_l1139_113911


namespace NUMINAMATH_CALUDE_evaluate_expression_l1139_113954

theorem evaluate_expression : -(20 / 2 * (6^2 + 10) - 120 + 5 * 6) = -370 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1139_113954


namespace NUMINAMATH_CALUDE_even_odd_function_sum_l1139_113941

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem even_odd_function_sum (f g : ℝ → ℝ) 
  (hf : is_even_function f) (hg : is_odd_function g) 
  (h : ∀ x, f x + g x = Real.exp x) : 
  ∀ x, g x = Real.exp x - Real.exp (-x) := by
  sorry

end NUMINAMATH_CALUDE_even_odd_function_sum_l1139_113941


namespace NUMINAMATH_CALUDE_point_on_bisector_l1139_113908

/-- 
Given a point (a, 2) in the second quadrant and on the angle bisector of the coordinate axes,
prove that a = -2.
-/
theorem point_on_bisector (a : ℝ) :
  (a < 0) →  -- Point is in the second quadrant
  (a = -2) →  -- Point is on the angle bisector
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_point_on_bisector_l1139_113908


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1139_113933

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) : ℝ) =
    (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = 3 ∧ B = -9 ∧ C = -9 ∧ D = 9 ∧ E = 165 ∧ F = 51 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1139_113933


namespace NUMINAMATH_CALUDE_craigs_remaining_apples_l1139_113951

/-- Calculates the number of apples Craig has after sharing -/
def craigs_apples_after_sharing (initial_apples : ℕ) (shared_apples : ℕ) : ℕ :=
  initial_apples - shared_apples

theorem craigs_remaining_apples :
  craigs_apples_after_sharing 20 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_craigs_remaining_apples_l1139_113951


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1139_113961

/-- Given that quantities a and b vary inversely, this function represents their relationship -/
def inverse_variation (k : ℝ) (a b : ℝ) : Prop := a * b = k

theorem inverse_variation_problem (k : ℝ) :
  inverse_variation k 800 0.5 →
  inverse_variation k 1600 0.25 :=
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1139_113961


namespace NUMINAMATH_CALUDE_parabola_directrix_l1139_113976

/-- Given a parabola with equation y² = 16x, its directrix has equation x = -4 -/
theorem parabola_directrix (x y : ℝ) : 
  (y^2 = 16*x) → (∃ p : ℝ, p = 4 ∧ x = -p) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1139_113976


namespace NUMINAMATH_CALUDE_original_fraction_l1139_113928

theorem original_fraction (x y : ℚ) : 
  x > 0 ∧ y > 0 →
  (120 / 100 * x) / (75 / 100 * y) = 2 / 15 →
  x / y = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_original_fraction_l1139_113928


namespace NUMINAMATH_CALUDE_police_can_see_bandit_l1139_113917

/-- Represents a point in the city grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a policeman -/
structure Policeman where
  position : Point
  canSeeInfinitely : Bool

/-- Represents the bandit -/
structure Bandit where
  position : Point

/-- Represents the city -/
structure City where
  grid : Set Point
  police : Set Policeman
  bandit : Bandit

/-- Represents the initial configuration of the city -/
def initialCity : City :=
  { grid := Set.univ,
    police := { p | ∃ k : ℤ, p.position = ⟨100 * k, 0⟩ ∧ p.canSeeInfinitely = true },
    bandit := ⟨⟨0, 0⟩⟩ }  -- Arbitrary initial position for the bandit

/-- Represents a strategy for the police -/
def PoliceStrategy := City → City

/-- Theorem: There exists a police strategy that guarantees seeing the bandit -/
theorem police_can_see_bandit :
  ∃ (strategy : PoliceStrategy), ∀ (c : City),
    ∃ (t : ℕ), ∃ (p : Policeman),
      p ∈ (strategy^[t] c).police ∧
      (strategy^[t] c).bandit.position.x = p.position.x ∨
      (strategy^[t] c).bandit.position.y = p.position.y :=
sorry

end NUMINAMATH_CALUDE_police_can_see_bandit_l1139_113917


namespace NUMINAMATH_CALUDE_larger_number_proof_l1139_113966

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : max x y = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1139_113966


namespace NUMINAMATH_CALUDE_test_score_result_l1139_113948

/-- Represents the score calculation for a test with specific conditions -/
def test_score (total_questions : ℕ) 
               (single_answer_questions : ℕ) 
               (multiple_answer_questions : ℕ) 
               (single_answer_marks : ℕ) 
               (multiple_answer_marks : ℕ) 
               (single_answer_penalty : ℕ) 
               (multiple_answer_penalty : ℕ) 
               (jose_wrong_single : ℕ) 
               (jose_wrong_multiple : ℕ) 
               (meghan_diff : ℕ) 
               (alisson_diff : ℕ) : ℕ := 
  sorry

theorem test_score_result : 
  test_score 70 50 20 2 4 1 2 10 5 30 50 = 280 :=
sorry

end NUMINAMATH_CALUDE_test_score_result_l1139_113948


namespace NUMINAMATH_CALUDE_fourth_root_squared_l1139_113972

theorem fourth_root_squared (x : ℝ) : (x^(1/4))^2 = 16 → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_squared_l1139_113972


namespace NUMINAMATH_CALUDE_arrangements_with_fixed_order_l1139_113996

/-- The number of programs --/
def total_programs : ℕ := 5

/-- The number of programs that must appear in a specific order --/
def fixed_order_programs : ℕ := 3

/-- The number of different arrangements when 3 specific programs must appear in a given order --/
def num_arrangements : ℕ := 20

/-- Theorem stating that given 5 programs with 3 in a fixed order, there are 20 different arrangements --/
theorem arrangements_with_fixed_order :
  total_programs = 5 →
  fixed_order_programs = 3 →
  num_arrangements = 20 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_with_fixed_order_l1139_113996


namespace NUMINAMATH_CALUDE_du_chin_pies_l1139_113967

/-- The number of meat pies Du Chin bakes in a day -/
def num_pies : ℕ := 200

/-- The price of each meat pie in dollars -/
def price_per_pie : ℕ := 20

/-- The fraction of sales used to buy ingredients for the next day -/
def ingredient_fraction : ℚ := 3/5

/-- The amount remaining after setting aside money for ingredients -/
def remaining_amount : ℕ := 1600

/-- Theorem stating that the number of pies baked satisfies the given conditions -/
theorem du_chin_pies :
  (num_pies * price_per_pie : ℚ) * (1 - ingredient_fraction) = remaining_amount := by
  sorry

end NUMINAMATH_CALUDE_du_chin_pies_l1139_113967


namespace NUMINAMATH_CALUDE_shortest_path_ratio_bound_l1139_113903

/-- Represents a city in the network -/
structure City where
  id : Nat

/-- Represents the road network -/
structure RoadNetwork where
  cities : Set City
  distance : City → City → ℝ
  shortest_path_length : City → ℝ

/-- The main theorem: the ratio of shortest path lengths between any two cities is at most 1.5 -/
theorem shortest_path_ratio_bound (network : RoadNetwork) :
  ∀ c1 c2 : City, c1 ∈ network.cities → c2 ∈ network.cities →
  network.shortest_path_length c1 ≤ 1.5 * network.shortest_path_length c2 :=
by sorry

end NUMINAMATH_CALUDE_shortest_path_ratio_bound_l1139_113903


namespace NUMINAMATH_CALUDE_smallest_x_floor_is_13_l1139_113919

-- Define the tangent function for degrees
noncomputable def tan_deg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- Define the property for x
def is_valid_x (x : ℝ) : Prop :=
  x > 2 ∧ tan_deg x = tan_deg (x^2)

-- State the theorem
theorem smallest_x_floor_is_13 :
  ∃ x : ℝ, is_valid_x x ∧ 
  (∀ y : ℝ, is_valid_x y → x ≤ y) ∧
  ⌊x⌋ = 13 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_floor_is_13_l1139_113919


namespace NUMINAMATH_CALUDE_tommy_initial_balloons_l1139_113925

/-- The number of balloons Tommy had initially -/
def initial_balloons : ℕ := 26

/-- The number of balloons Tommy's mom gave him -/
def mom_balloons : ℕ := 34

/-- The total number of balloons Tommy had after receiving more from his mom -/
def total_balloons : ℕ := 60

/-- Theorem: Tommy had 26 balloons to start with -/
theorem tommy_initial_balloons : 
  initial_balloons + mom_balloons = total_balloons :=
by sorry

end NUMINAMATH_CALUDE_tommy_initial_balloons_l1139_113925


namespace NUMINAMATH_CALUDE_min_grades_for_average_l1139_113989

theorem min_grades_for_average (n : ℕ) (s : ℕ) : n ≥ 51 ↔ 
  (∃ s : ℕ, (4.5 : ℝ) < (s : ℝ) / (n : ℝ) ∧ (s : ℝ) / (n : ℝ) < 4.51) :=
sorry

end NUMINAMATH_CALUDE_min_grades_for_average_l1139_113989


namespace NUMINAMATH_CALUDE_soda_per_syrup_box_l1139_113939

/-- Given a convenience store that sells soda and buys syrup boxes, this theorem proves
    the number of gallons of soda that can be made from one box of syrup. -/
theorem soda_per_syrup_box 
  (total_soda : ℝ) 
  (box_cost : ℝ) 
  (total_syrup_cost : ℝ) 
  (h1 : total_soda = 180) 
  (h2 : box_cost = 40) 
  (h3 : total_syrup_cost = 240) : 
  total_soda / (total_syrup_cost / box_cost) = 30 := by
sorry

end NUMINAMATH_CALUDE_soda_per_syrup_box_l1139_113939


namespace NUMINAMATH_CALUDE_perimeter_ratio_of_squares_l1139_113970

theorem perimeter_ratio_of_squares (s1 s2 : Real) (h : s1^2 / s2^2 = 16 / 25) :
  (4 * s1) / (4 * s2) = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_perimeter_ratio_of_squares_l1139_113970


namespace NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1139_113968

/-- The number of different rectangles in a rectangular grid --/
def num_rectangles (rows : ℕ) (cols : ℕ) : ℕ :=
  (rows.choose 2) * (cols.choose 2)

/-- Theorem: In a 5x4 grid, the number of different rectangles is 60 --/
theorem rectangles_in_5x4_grid :
  num_rectangles 5 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1139_113968


namespace NUMINAMATH_CALUDE_min_value_xy_expression_min_value_achievable_l1139_113902

theorem min_value_xy_expression (x y : ℝ) : (x * y + 1)^2 + (x - y)^2 ≥ 1 := by sorry

theorem min_value_achievable : ∃ x y : ℝ, (x * y + 1)^2 + (x - y)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_min_value_xy_expression_min_value_achievable_l1139_113902


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_l1139_113913

theorem opposite_numbers_sum (a b : ℤ) : (a + b = 0) → (2006 * a + 2006 * b = 0) := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_l1139_113913


namespace NUMINAMATH_CALUDE_probability_is_one_third_l1139_113985

/-- Represents a ball with a label -/
structure Ball :=
  (label : ℕ)

/-- The set of all balls in the box -/
def box : Finset Ball := sorry

/-- The condition that the sum of labels on two balls is 5 -/
def sumIs5 (b1 b2 : Ball) : Prop :=
  b1.label + b2.label = 5

/-- The set of all possible pairs of balls -/
def allPairs : Finset (Ball × Ball) := sorry

/-- The set of favorable pairs (sum is 5) -/
def favorablePairs : Finset (Ball × Ball) := sorry

/-- The probability of drawing two balls with sum 5 -/
def probability : ℚ := (favorablePairs.card : ℚ) / (allPairs.card : ℚ)

theorem probability_is_one_third :
  probability = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l1139_113985


namespace NUMINAMATH_CALUDE_sally_bread_consumption_l1139_113999

/-- The number of sandwiches Sally eats on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The number of sandwiches Sally eats on Sunday -/
def sunday_sandwiches : ℕ := 1

/-- The number of pieces of bread used in each sandwich -/
def bread_per_sandwich : ℕ := 2

/-- The total number of pieces of bread Sally eats across Saturday and Sunday -/
def total_bread : ℕ := (saturday_sandwiches + sunday_sandwiches) * bread_per_sandwich

theorem sally_bread_consumption :
  total_bread = 6 :=
by sorry

end NUMINAMATH_CALUDE_sally_bread_consumption_l1139_113999
