import Mathlib

namespace parallel_perpendicular_implication_parallel_planes_perpendicular_implication_l3760_376063

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Theorem 1
theorem parallel_perpendicular_implication
  (m n : Line) (α : Plane)
  (h1 : parallel_lines m n)
  (h2 : perpendicular m α) :
  perpendicular n α :=
sorry

-- Theorem 2
theorem parallel_planes_perpendicular_implication
  (m n : Line) (α β : Plane)
  (h1 : parallel_planes α β)
  (h2 : parallel_lines m n)
  (h3 : perpendicular m α) :
  perpendicular n β :=
sorry

end parallel_perpendicular_implication_parallel_planes_perpendicular_implication_l3760_376063


namespace sachins_age_l3760_376033

theorem sachins_age (rahuls_age : ℝ) : 
  (rahuls_age + 7) / rahuls_age = 11 / 9 → rahuls_age + 7 = 38.5 := by
  sorry

end sachins_age_l3760_376033


namespace polynomial_equality_l3760_376091

theorem polynomial_equality : 105^5 - 5 * 105^4 + 10 * 105^3 - 10 * 105^2 + 5 * 105 - 1 = 11714628224 := by
  sorry

end polynomial_equality_l3760_376091


namespace diamond_calculation_l3760_376048

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- State the theorem
theorem diamond_calculation :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29/132 := by
  sorry

end diamond_calculation_l3760_376048


namespace special_polynomial_form_l3760_376013

/-- A polynomial in two variables satisfying specific conditions -/
structure SpecialPolynomial where
  P : ℝ → ℝ → ℝ
  n : ℕ+
  homogeneous : ∀ (t x y : ℝ), P (t * x) (t * y) = t ^ n.val * P x y
  cyclic_sum : ∀ (x y z : ℝ), P (y + z) x + P (z + x) y + P (x + y) z = 0
  normalization : P 1 0 = 1

/-- The theorem stating the form of the special polynomial -/
theorem special_polynomial_form (sp : SpecialPolynomial) :
  ∀ (x y : ℝ), sp.P x y = (x + y) ^ sp.n.val * (x - 2 * y) := by
  sorry

end special_polynomial_form_l3760_376013


namespace certain_number_problem_l3760_376038

theorem certain_number_problem : ∃ x : ℕ, 
  220040 = (x + 445) * (2 * (x - 445)) + 40 ∧ x = 555 := by
  sorry

end certain_number_problem_l3760_376038


namespace phika_inequality_l3760_376080

/-- A sextuple of positive real numbers is phika if the sum of a's equals the sum of b's equals 1 -/
def IsPhika (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) : Prop :=
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0 ∧
  a₁ + a₂ + a₃ = 1 ∧ b₁ + b₂ + b₃ = 1

theorem phika_inequality :
  (∃ a₁ a₂ a₃ b₁ b₂ b₃ : ℝ, IsPhika a₁ a₂ a₃ b₁ b₂ b₃ ∧
    a₁ * (Real.sqrt b₁ + a₂) + a₂ * (Real.sqrt b₂ + a₃) + a₃ * (Real.sqrt b₃ + a₁) > 1 - 1 / (2022^2022)) ∧
  (∀ a₁ a₂ a₃ b₁ b₂ b₃ : ℝ, IsPhika a₁ a₂ a₃ b₁ b₂ b₃ →
    a₁ * (Real.sqrt b₁ + a₂) + a₂ * (Real.sqrt b₂ + a₃) + a₃ * (Real.sqrt b₃ + a₁) < 1) := by
  sorry

end phika_inequality_l3760_376080


namespace pedro_gifts_l3760_376082

theorem pedro_gifts (total : ℕ) (emilio : ℕ) (jorge : ℕ) 
  (h1 : total = 21)
  (h2 : emilio = 11)
  (h3 : jorge = 6) :
  total - (emilio + jorge) = 4 := by
  sorry

end pedro_gifts_l3760_376082


namespace paper_used_l3760_376005

theorem paper_used (initial : ℕ) (remaining : ℕ) (used : ℕ) 
  (h1 : initial = 900) 
  (h2 : remaining = 744) 
  (h3 : used = initial - remaining) : used = 156 := by
  sorry

end paper_used_l3760_376005


namespace answer_key_combinations_l3760_376086

/-- Represents the number of answer choices for a multiple-choice question -/
def multipleChoiceOptions : ℕ := 4

/-- Represents the number of true-false questions -/
def trueFalseQuestions : ℕ := 3

/-- Represents the number of multiple-choice questions -/
def multipleChoiceQuestions : ℕ := 2

/-- Calculates the number of valid true-false combinations -/
def validTrueFalseCombinations : ℕ := 2^trueFalseQuestions - 2

/-- Calculates the number of multiple-choice combinations -/
def multipleChoiceCombinations : ℕ := multipleChoiceOptions^multipleChoiceQuestions

/-- Theorem stating the total number of ways to create the answer key -/
theorem answer_key_combinations :
  validTrueFalseCombinations * multipleChoiceCombinations = 96 := by
  sorry

end answer_key_combinations_l3760_376086


namespace allysons_age_l3760_376050

theorem allysons_age (hirams_age allyson_age : ℕ) : 
  hirams_age = 40 →
  hirams_age + 12 = 2 * allyson_age - 4 →
  allyson_age = 28 := by
sorry

end allysons_age_l3760_376050


namespace two_digit_swap_l3760_376085

/-- 
Given a two-digit number with 1 in the tens place and x in the ones place,
if swapping these digits results in a number 18 greater than the original,
then the equation 10x + 1 - (10 + x) = 18 holds.
-/
theorem two_digit_swap (x : ℕ) : 
  (x < 10) →  -- Ensure x is a single digit
  (10 * x + 1) - (10 + x) = 18 := by
  sorry

end two_digit_swap_l3760_376085


namespace solution_set_equals_interval_l3760_376076

theorem solution_set_equals_interval :
  {x : ℝ | x ≤ 1} = Set.Iic 1 := by sorry

end solution_set_equals_interval_l3760_376076


namespace convention_handshakes_l3760_376045

/-- Represents the convention of twins and triplets --/
structure Convention where
  twin_sets : ℕ
  triplet_sets : ℕ

/-- Calculates the total number of handshakes in the convention --/
def total_handshakes (c : Convention) : ℕ :=
  let twin_count := c.twin_sets * 2
  let triplet_count := c.triplet_sets * 3
  let twin_handshakes := (twin_count * (twin_count - 2)) / 2
  let triplet_handshakes := (triplet_count * (triplet_count - 3)) / 2
  let twin_to_triplet := twin_count * (triplet_count / 2)
  twin_handshakes + triplet_handshakes + twin_to_triplet

/-- The theorem stating that the total number of handshakes in the given convention is 354 --/
theorem convention_handshakes :
  total_handshakes ⟨10, 4⟩ = 354 := by sorry

end convention_handshakes_l3760_376045


namespace arithmetic_sequence_logarithm_l3760_376027

theorem arithmetic_sequence_logarithm (x : ℝ) : 
  (∃ r : ℝ, Real.log 2 + r = Real.log (2^x - 1) ∧ 
             Real.log (2^x - 1) + r = Real.log (2^x + 3)) → 
  x = Real.log 5 / Real.log 2 := by
sorry

end arithmetic_sequence_logarithm_l3760_376027


namespace triple_hash_70_approx_8_l3760_376059

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.4 * N + 2

-- State the theorem
theorem triple_hash_70_approx_8 : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |hash (hash (hash 70)) - 8| < ε :=
sorry

end triple_hash_70_approx_8_l3760_376059


namespace closed_triangular_path_steps_divisible_by_three_l3760_376065

/-- A closed path on a triangular lattice -/
structure TriangularPath where
  steps : ℕ
  is_closed : Bool

/-- Theorem: The number of steps in a closed path on a triangular lattice is divisible by 3 -/
theorem closed_triangular_path_steps_divisible_by_three (path : TriangularPath) 
  (h : path.is_closed = true) : 
  ∃ k : ℕ, path.steps = 3 * k := by
  sorry

end closed_triangular_path_steps_divisible_by_three_l3760_376065


namespace train_length_l3760_376046

/-- Calculates the length of a train given its speed and time to cross a post -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 40 → time = 17.1 → speed * time * (5 / 18) = 190 := by
  sorry

end train_length_l3760_376046


namespace apple_count_theorem_l3760_376016

def is_valid_apple_count (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ (∃ k : ℕ, n = 6 * k)

theorem apple_count_theorem : 
  ∀ n : ℕ, is_valid_apple_count n ↔ (n = 72 ∨ n = 78) :=
by sorry

end apple_count_theorem_l3760_376016


namespace opposite_of_negative_fraction_l3760_376056

theorem opposite_of_negative_fraction :
  ∀ (x : ℚ), x = -6/7 → (x + 6/7 = 0) :=
by
  sorry

end opposite_of_negative_fraction_l3760_376056


namespace trip_duration_l3760_376084

theorem trip_duration (duration_first : ℝ) 
  (h1 : duration_first ≥ 0)
  (h2 : duration_first + 2 * duration_first + 2 * duration_first = 10) :
  duration_first = 2 := by
sorry

end trip_duration_l3760_376084


namespace sum_of_a_and_b_l3760_376040

theorem sum_of_a_and_b (a b : ℝ) : 
  (a + Real.sqrt b + (a - Real.sqrt b) = -6) →
  ((a + Real.sqrt b) * (a - Real.sqrt b) = 4) →
  a + b = 2 := by
sorry

end sum_of_a_and_b_l3760_376040


namespace line_equations_l3760_376035

-- Define the lines m and n
def line_m (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line_n (x y : ℝ) : Prop := x + y - 3 = 0

-- Define point P as the intersection of m and n
def point_P : ℝ × ℝ := (1, 2)

-- Define points A and B
def point_A : ℝ × ℝ := (1, 3)
def point_B : ℝ × ℝ := (3, 2)

-- Define line l
def line_l (x y : ℝ) : Prop := (x + 2 * y - 4 = 0) ∨ (x = 2)

-- Define line l₁
def line_l1 (x y : ℝ) : Prop := y = -1/2 * x + 2

-- State the theorem
theorem line_equations :
  (∀ x y : ℝ, line_m x y ∧ line_n x y → (x, y) = point_P) →
  (∀ x y : ℝ, line_l x y → (x, y) = point_P) →
  (∀ x y : ℝ, line_l1 x y → (x, y) = point_P) →
  (∀ x y : ℝ, line_l x y → 
    abs ((2*x - 2*point_A.1 + y - point_A.2) / Real.sqrt (5)) = 
    abs ((2*x - 2*point_B.1 + y - point_B.2) / Real.sqrt (5))) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    (∀ x y : ℝ, line_l1 x y ↔ x/a + y/b = 1) ∧
    1/2 * a * b = 4) →
  (∀ x y : ℝ, line_l x y ∨ line_l1 x y) :=
sorry

end line_equations_l3760_376035


namespace jeff_saturday_laps_l3760_376094

theorem jeff_saturday_laps (total_laps : ℕ) (sunday_morning_laps : ℕ) (remaining_laps : ℕ) 
  (h1 : total_laps = 98)
  (h2 : sunday_morning_laps = 15)
  (h3 : remaining_laps = 56) :
  total_laps - (sunday_morning_laps + remaining_laps) = 27 := by
  sorry

end jeff_saturday_laps_l3760_376094


namespace set_intersection_equality_l3760_376047

theorem set_intersection_equality (S T : Set ℝ) : 
  S = {y | ∃ x, y = (3 : ℝ) ^ x} →
  T = {y | ∃ x, y = x^2 + 1} →
  S ∩ T = T := by sorry

end set_intersection_equality_l3760_376047


namespace two_solutions_iff_a_gt_one_third_l3760_376074

/-- The equation |x-3| = ax - 1 has two solutions if and only if a > 1/3 -/
theorem two_solutions_iff_a_gt_one_third (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (|x₁ - 3| = a * x₁ - 1) ∧ (|x₂ - 3| = a * x₂ - 1)) ↔ a > 1/3 := by
  sorry

end two_solutions_iff_a_gt_one_third_l3760_376074


namespace complex_modulus_problem_l3760_376058

theorem complex_modulus_problem (z : ℂ) (h : z * (2 + Complex.I) = 5 * Complex.I - 10) : 
  Complex.abs z = 5 := by
  sorry

end complex_modulus_problem_l3760_376058


namespace problem_solution_l3760_376095

-- Define the conditions
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 3*x - 10 > 0

-- Define the theorem
theorem problem_solution :
  (∀ a : ℝ, a > 0) →
  (∀ x : ℝ, p x 1 ∧ q x → 2 < x ∧ x < 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, ¬(p x a) → ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x a) → 1 < a ∧ a ≤ 2) :=
sorry

end problem_solution_l3760_376095


namespace binder_cost_l3760_376067

theorem binder_cost (book_cost : ℕ) (num_binders : ℕ) (num_notebooks : ℕ) 
  (notebook_cost : ℕ) (total_cost : ℕ) : ℕ :=
by
  have h1 : book_cost = 16 := by sorry
  have h2 : num_binders = 3 := by sorry
  have h3 : num_notebooks = 6 := by sorry
  have h4 : notebook_cost = 1 := by sorry
  have h5 : total_cost = 28 := by sorry
  
  have binder_cost : ℕ := (total_cost - (book_cost + num_notebooks * notebook_cost)) / num_binders
  
  exact binder_cost

end binder_cost_l3760_376067


namespace apps_after_deletion_l3760_376057

/-- Represents the number of apps on Faye's phone. -/
structure PhoneApps where
  total : ℕ
  gaming : ℕ
  utility : ℕ
  gaming_deleted : ℕ
  utility_deleted : ℕ

/-- Calculates the number of remaining apps after deletion. -/
def remaining_apps (apps : PhoneApps) : ℕ :=
  apps.total - (apps.gaming_deleted + apps.utility_deleted)

/-- Theorem stating the number of remaining apps after deletion. -/
theorem apps_after_deletion (apps : PhoneApps)
  (h1 : apps.total = 12)
  (h2 : apps.gaming = 5)
  (h3 : apps.utility = apps.total - apps.gaming)
  (h4 : apps.gaming_deleted = 4)
  (h5 : apps.utility_deleted = 3)
  (h6 : apps.gaming - apps.gaming_deleted ≥ 1)
  (h7 : apps.utility - apps.utility_deleted ≥ 1) :
  remaining_apps apps = 5 := by
  sorry


end apps_after_deletion_l3760_376057


namespace twenty_fifth_number_l3760_376099

def twisted_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- We define 0th term as 0 for convenience
  | 1 => 1  -- First term is 1
  | n + 1 => 
    if n % 5 = 0 then twisted_sequence n + 1  -- Every 6th number (5th index) is previous + 1
    else 2 * twisted_sequence n  -- Otherwise, double the previous number

theorem twenty_fifth_number : twisted_sequence 25 = 69956 := by
  sorry

end twenty_fifth_number_l3760_376099


namespace prob_diff_absolute_l3760_376068

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1200

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 800

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def prob_same_color : ℚ :=
  (red_marbles.choose 2 + black_marbles.choose 2) / total_marbles.choose 2

/-- The probability of drawing two marbles of different colors -/
def prob_diff_color : ℚ :=
  (red_marbles * black_marbles) / total_marbles.choose 2

/-- Theorem: The absolute difference between the probability of drawing two marbles
    of the same color and the probability of drawing two marbles of different colors
    is 789/19990 -/
theorem prob_diff_absolute : |prob_same_color - prob_diff_color| = 789 / 19990 := by
  sorry

end prob_diff_absolute_l3760_376068


namespace push_ups_total_l3760_376015

theorem push_ups_total (david_pushups : ℕ) (difference : ℕ) : 
  david_pushups = 51 → difference = 49 → 
  david_pushups + (david_pushups - difference) = 53 := by
  sorry

end push_ups_total_l3760_376015


namespace melanie_balloons_l3760_376044

theorem melanie_balloons (joan_balloons total_balloons : ℕ) 
  (h1 : joan_balloons = 40)
  (h2 : total_balloons = 81) :
  total_balloons - joan_balloons = 41 :=
by
  sorry

end melanie_balloons_l3760_376044


namespace walnut_trees_after_planting_l3760_376036

/-- The number of walnut trees in the park after planting -/
def trees_after_planting (initial_trees newly_planted_trees : ℕ) : ℕ :=
  initial_trees + newly_planted_trees

/-- Theorem: The number of walnut trees in the park after planting is 77 -/
theorem walnut_trees_after_planting :
  trees_after_planting 22 55 = 77 := by
  sorry

end walnut_trees_after_planting_l3760_376036


namespace music_school_population_l3760_376000

/-- Given a music school with boys, girls, and teachers, prove that the total number of people is 9b/7, where b is the number of boys. -/
theorem music_school_population (b g t : ℚ) : 
  b = 4 * g ∧ g = 7 * t → b + g + t = 9 * b / 7 :=
by
  sorry

end music_school_population_l3760_376000


namespace maud_olive_flea_multiple_l3760_376055

/-- The number of fleas on Gertrude -/
def gertrude_fleas : ℕ := 10

/-- The number of fleas on Olive -/
def olive_fleas : ℕ := gertrude_fleas / 2

/-- The total number of fleas on all chickens -/
def total_fleas : ℕ := 40

/-- The number of fleas on Maud -/
def maud_fleas : ℕ := total_fleas - gertrude_fleas - olive_fleas

/-- The multiple of fleas Maud has compared to Olive -/
def maud_olive_multiple : ℕ := maud_fleas / olive_fleas

theorem maud_olive_flea_multiple :
  maud_olive_multiple = 5 := by sorry

end maud_olive_flea_multiple_l3760_376055


namespace soup_per_bag_is_three_l3760_376009

-- Define the quantities
def milk_quarts : ℚ := 2
def vegetable_quarts : ℚ := 1
def num_bags : ℕ := 3

-- Define the relationship between milk and chicken stock
def chicken_stock_quarts : ℚ := 3 * milk_quarts

-- Calculate the total amount of soup
def total_soup : ℚ := milk_quarts + chicken_stock_quarts + vegetable_quarts

-- Define the amount of soup per bag
def soup_per_bag : ℚ := total_soup / num_bags

-- Theorem to prove
theorem soup_per_bag_is_three : soup_per_bag = 3 := by
  sorry

end soup_per_bag_is_three_l3760_376009


namespace probability_three_white_balls_l3760_376004

def total_balls : ℕ := 15
def white_balls : ℕ := 7
def black_balls : ℕ := 8
def drawn_balls : ℕ := 3

theorem probability_three_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 13 := by
  sorry

end probability_three_white_balls_l3760_376004


namespace cookies_calculation_l3760_376090

/-- The number of people Brenda's mother made cookies for -/
def num_people : ℕ := 14

/-- The number of cookies each person had -/
def cookies_per_person : ℕ := 30

/-- The total number of cookies prepared -/
def total_cookies : ℕ := num_people * cookies_per_person

theorem cookies_calculation : total_cookies = 420 := by
  sorry

end cookies_calculation_l3760_376090


namespace min_value_problem_l3760_376060

theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (1 / a + 1 / b + 1 / c ≥ 9) ∧ (1 / (3 * a + 2) + 1 / (3 * b + 2) + 1 / (3 * c + 2) ≥ 1) := by
  sorry

end min_value_problem_l3760_376060


namespace probability_of_no_mismatch_l3760_376096

/-- The number of red socks -/
def num_red_socks : ℕ := 4

/-- The number of blue socks -/
def num_blue_socks : ℕ := 4

/-- The total number of socks -/
def total_socks : ℕ := num_red_socks + num_blue_socks

/-- The number of pairs to be formed -/
def num_pairs : ℕ := total_socks / 2

/-- The number of ways to divide red socks into pairs -/
def red_pairings : ℕ := (Nat.choose num_red_socks 2) / 2

/-- The number of ways to divide blue socks into pairs -/
def blue_pairings : ℕ := (Nat.choose num_blue_socks 2) / 2

/-- The total number of favorable pairings -/
def favorable_pairings : ℕ := red_pairings * blue_pairings

/-- The total number of possible pairings -/
def total_pairings : ℕ := (Nat.factorial total_socks) / ((Nat.factorial 2)^num_pairs * Nat.factorial num_pairs)

/-- The probability of no mismatched pairs -/
def probability_no_mismatch : ℚ := favorable_pairings / total_pairings

theorem probability_of_no_mismatch : probability_no_mismatch = 3 / 35 := by
  sorry

end probability_of_no_mismatch_l3760_376096


namespace boat_speed_in_still_water_l3760_376093

/-- Proves that the speed of a boat in still water is 16 km/hr given specific downstream conditions. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 6)
  (h3 : downstream_distance = 126) :
  downstream_distance = (boat_speed + stream_speed) * downstream_time → 
  boat_speed = 16 :=
by
  sorry

#check boat_speed_in_still_water

end boat_speed_in_still_water_l3760_376093


namespace octahedron_colorings_l3760_376052

/-- The number of faces in a regular octahedron -/
def num_faces : ℕ := 8

/-- The number of rotational symmetries of a regular octahedron -/
def num_rotational_symmetries : ℕ := 24

/-- The number of distinguishable colorings of a regular octahedron -/
def num_distinguishable_colorings : ℕ := Nat.factorial num_faces / num_rotational_symmetries

theorem octahedron_colorings :
  num_distinguishable_colorings = 1680 := by sorry

end octahedron_colorings_l3760_376052


namespace geometric_sequence_property_l3760_376088

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem statement -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  GeometricSequence a →
  a 3 * a 5 * a 7 * a 9 * a 11 = 243 →
  a 9 ^ 2 / a 11 = 3 := by
  sorry

end geometric_sequence_property_l3760_376088


namespace baby_whales_count_l3760_376006

/-- Represents the number of whales observed during Ishmael's monitoring --/
structure WhaleCount where
  first_trip_males : ℕ
  first_trip_females : ℕ
  third_trip_males : ℕ
  third_trip_females : ℕ
  total_whales : ℕ

/-- Theorem stating the number of baby whales observed on the second trip --/
theorem baby_whales_count (w : WhaleCount) 
  (h1 : w.first_trip_males = 28)
  (h2 : w.first_trip_females = 2 * w.first_trip_males)
  (h3 : w.third_trip_males = w.first_trip_males / 2)
  (h4 : w.third_trip_females = w.first_trip_females)
  (h5 : w.total_whales = 178) :
  w.total_whales - (w.first_trip_males + w.first_trip_females + w.third_trip_males + w.third_trip_females) = 24 := by
  sorry

end baby_whales_count_l3760_376006


namespace inverse_f_at_407_l3760_376023

noncomputable def f (x : ℝ) : ℝ := 5 * x^4 + 2

theorem inverse_f_at_407 : Function.invFun f 407 = 3 := by sorry

end inverse_f_at_407_l3760_376023


namespace mans_rowing_speed_l3760_376069

/-- Man's rowing problem with wind resistance -/
theorem mans_rowing_speed (upstream_speed downstream_speed wind_effect : ℝ) 
  (h1 : upstream_speed = 25)
  (h2 : downstream_speed = 45)
  (h3 : wind_effect = 2) :
  let still_water_speed := (upstream_speed + downstream_speed) / 2
  let adjusted_upstream_speed := upstream_speed - wind_effect
  let adjusted_downstream_speed := downstream_speed + wind_effect
  let adjusted_still_water_speed := (adjusted_upstream_speed + adjusted_downstream_speed) / 2
  adjusted_still_water_speed = still_water_speed :=
by sorry

end mans_rowing_speed_l3760_376069


namespace direct_sort_5_rounds_l3760_376017

def initial_sequence : List Nat := [49, 38, 65, 97, 76, 13, 27]

def direct_sort_step (l : List Nat) : List Nat :=
  match l with
  | [] => []
  | _ => let max := l.maximum? |>.getD 0
         max :: (l.filter (· ≠ max))

def direct_sort (l : List Nat) (n : Nat) : List Nat :=
  match n with
  | 0 => l
  | n + 1 => direct_sort (direct_sort_step l) n

theorem direct_sort_5_rounds :
  direct_sort initial_sequence 5 = [97, 76, 65, 49, 38, 13, 27] := by
  sorry

end direct_sort_5_rounds_l3760_376017


namespace teaching_arrangements_count_l3760_376029

def number_of_teachers : ℕ := 3
def number_of_classes : ℕ := 6
def classes_per_teacher : ℕ := 2

theorem teaching_arrangements_count :
  (Nat.choose number_of_classes classes_per_teacher) *
  (Nat.choose (number_of_classes - classes_per_teacher) classes_per_teacher) *
  (Nat.choose (number_of_classes - 2 * classes_per_teacher) classes_per_teacher) = 90 := by
  sorry

end teaching_arrangements_count_l3760_376029


namespace quadratic_equation_roots_l3760_376007

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 2*k*x^2 + (8*k+1)*x + 8*k = 0 ∧ 2*k*y^2 + (8*k+1)*y + 8*k = 0) 
  ↔ 
  (k ≥ -1/16 ∧ k ≠ 0) :=
sorry

end quadratic_equation_roots_l3760_376007


namespace kate_age_l3760_376077

theorem kate_age (total_age : ℕ) (maggie_age : ℕ) (sue_age : ℕ) 
  (h1 : total_age = 48)
  (h2 : maggie_age = 17)
  (h3 : sue_age = 12) :
  total_age - maggie_age - sue_age = 19 := by
  sorry

end kate_age_l3760_376077


namespace functional_equation_properties_l3760_376018

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem functional_equation_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (f 0 = 0) ∧ (f 1 = 0) ∧ (∀ x : ℝ, f (-x) = f x) := by
  sorry

end functional_equation_properties_l3760_376018


namespace items_sold_l3760_376014

/-- Given the following conditions:
  1. A grocery store ordered 4458 items to restock.
  2. They have 575 items in the storeroom.
  3. They have 3,472 items left in the whole store.
  Prove that the number of items sold that day is 1561. -/
theorem items_sold (restocked : ℕ) (in_storeroom : ℕ) (left_in_store : ℕ) 
  (h1 : restocked = 4458)
  (h2 : in_storeroom = 575)
  (h3 : left_in_store = 3472) :
  restocked + in_storeroom - left_in_store = 1561 :=
by sorry

end items_sold_l3760_376014


namespace exists_infinite_subset_with_constant_gcd_l3760_376070

-- Define the set of natural numbers that are products of at most 1990 primes
def ProductOfLimitedPrimes (n : ℕ) : Prop :=
  ∃ (primes : Finset ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ primes.card ≤ 1990 ∧ n = primes.prod id

-- Define the property of A
def InfiniteSetOfLimitedPrimeProducts (A : Set ℕ) : Prop :=
  Set.Infinite A ∧ ∀ a ∈ A, ProductOfLimitedPrimes a

-- The main theorem
theorem exists_infinite_subset_with_constant_gcd
  (A : Set ℕ) (hA : InfiniteSetOfLimitedPrimeProducts A) :
  ∃ (B : Set ℕ) (k : ℕ), Set.Infinite B ∧ B ⊆ A ∧
    ∀ (x y : ℕ), x ∈ B → y ∈ B → x ≠ y → Nat.gcd x y = k :=
sorry

end exists_infinite_subset_with_constant_gcd_l3760_376070


namespace power_8_2048_mod_50_l3760_376043

theorem power_8_2048_mod_50 : 8^2048 % 50 = 38 := by sorry

end power_8_2048_mod_50_l3760_376043


namespace eight_people_arrangements_l3760_376039

/-- The number of ways to arrange n distinct objects in a line -/
def linearArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- There are 8! ways to arrange 8 people in a line -/
theorem eight_people_arrangements : linearArrangements 8 = 40320 := by
  sorry

end eight_people_arrangements_l3760_376039


namespace min_distance_point_l3760_376051

noncomputable def f (a x : ℝ) : ℝ := (x - a)^2 + (2 * Real.log x - 2 * a)^2

theorem min_distance_point (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ ≤ 4/5) → a = 1/5 := by
  sorry

end min_distance_point_l3760_376051


namespace total_area_is_135_l3760_376024

/-- Represents the geometry of villages, roads, fields, and forest --/
structure VillageGeometry where
  /-- Side length of the square field --/
  r : ℝ
  /-- Side length of the rectangular field along the road --/
  p : ℝ
  /-- Side length of the rectangular forest along the road --/
  q : ℝ

/-- The total area of the forest and fields is 135 sq km --/
theorem total_area_is_135 (g : VillageGeometry) : 
  g.r^2 + 4 * g.p^2 + 12 * g.q = 135 :=
by
  sorry

/-- The forest area is 45 sq km more than the sum of field areas --/
axiom forest_area_relation (g : VillageGeometry) : 
  12 * g.q = g.r^2 + 4 * g.p^2 + 45

/-- The side of the rectangular field perpendicular to the road is 4 times longer --/
axiom rectangular_field_proportion (g : VillageGeometry) : 
  4 * g.p = g.q

/-- The side of the rectangular forest perpendicular to the road is 12 km --/
axiom forest_width (g : VillageGeometry) : g.q = 12

end total_area_is_135_l3760_376024


namespace incorrect_step_l3760_376071

theorem incorrect_step (a b : ℝ) (h : a < b) : ¬(2 * (a - b)^2 < (a - b)^2) := by
  sorry

end incorrect_step_l3760_376071


namespace incorrect_addition_statement_l3760_376064

theorem incorrect_addition_statement : 
  (8 + 34 ≠ 32) ∧ (17 + 17 = 34) ∧ (15 + 13 = 28) := by
  sorry

end incorrect_addition_statement_l3760_376064


namespace three_digit_number_theorem_l3760_376053

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Checks if a tuple represents a valid three-digit number -/
def isValidThreeDigitNumber (n : ThreeDigitNumber) : Prop :=
  let (a, b, c) := n
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9

/-- Converts a three-digit number tuple to its numerical value -/
def toNumber (n : ThreeDigitNumber) : Nat :=
  let (a, b, c) := n
  100 * a + 10 * b + c

/-- Generates all permutations of a three-digit number -/
def permutations (n : ThreeDigitNumber) : List ThreeDigitNumber :=
  let (a, b, c) := n
  [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]

/-- Calculates the average of the permutations of a three-digit number -/
def averageOfPermutations (n : ThreeDigitNumber) : Nat :=
  (List.sum (List.map toNumber (permutations n))) / 6

/-- Checks if a three-digit number satisfies the given condition -/
def satisfiesCondition (n : ThreeDigitNumber) : Prop :=
  isValidThreeDigitNumber n ∧ averageOfPermutations n = toNumber n

/-- The set of three-digit numbers that satisfy the condition -/
def solutionSet : Set Nat :=
  {370, 407, 481, 518, 592, 629}

/-- The main theorem to be proved -/
theorem three_digit_number_theorem (n : ThreeDigitNumber) :
  satisfiesCondition n ↔ toNumber n ∈ solutionSet := by
  sorry

end three_digit_number_theorem_l3760_376053


namespace last_three_digits_are_218_l3760_376078

/-- A function that generates the list of positive integers starting with 2 -/
def digitsStartingWith2 (n : ℕ) : ℕ :=
  if n < 10 then 2
  else if n < 100 then 20 + (n - 10)
  else if n < 1000 then 200 + (n - 100)
  else 2000 + (n - 1000)

/-- A function that returns the nth digit in the list -/
def nthDigit (n : ℕ) : ℕ :=
  let number := digitsStartingWith2 ((n - 1) / 4 + 1)
  let digitPosition := (n - 1) % 4
  (number / (10 ^ (3 - digitPosition))) % 10

/-- The theorem to be proved -/
theorem last_three_digits_are_218 :
  (nthDigit 1198) * 100 + (nthDigit 1199) * 10 + nthDigit 1200 = 218 := by
  sorry


end last_three_digits_are_218_l3760_376078


namespace chessboard_uniquely_determined_l3760_376025

/-- Represents a chessboard with numbers 1 to 64 -/
def Chessboard := Fin 8 → Fin 8 → Fin 64

/-- The sum of numbers in a rectangle of two cells -/
def RectangleSum (board : Chessboard) (r1 c1 r2 c2 : Fin 8) : ℕ :=
  (board r1 c1).val + 1 + (board r2 c2).val + 1

/-- Predicate to check if two positions are on the same diagonal -/
def OnSameDiagonal (r1 c1 r2 c2 : Fin 8) : Prop :=
  r1 + c1 = r2 + c2 ∨ r1 + c2 = r2 + c1

/-- Main theorem -/
theorem chessboard_uniquely_determined 
  (board : Chessboard) 
  (sums_known : ∀ (r1 c1 r2 c2 : Fin 8), r1 = r2 ∧ c1.val + 1 = c2.val ∨ r1.val + 1 = r2.val ∧ c1 = c2 → 
    ∃ (s : ℕ), s = RectangleSum board r1 c1 r2 c2)
  (one_and_sixtyfour_on_diagonal : ∃ (r1 c1 r2 c2 : Fin 8), 
    board r1 c1 = 0 ∧ board r2 c2 = 63 ∧ OnSameDiagonal r1 c1 r2 c2) :
  ∀ (r c : Fin 8), ∃! (n : Fin 64), board r c = n :=
sorry

end chessboard_uniquely_determined_l3760_376025


namespace left_handed_mouse_price_increase_l3760_376097

/-- Represents the store's weekly operation --/
structure StoreOperation where
  daysOpen : Nat
  miceSoldPerDay : Nat
  normalMousePrice : Nat
  weeklyRevenue : Nat

/-- Calculates the percentage increase in price --/
def percentageIncrease (normalPrice leftHandedPrice : Nat) : Nat :=
  ((leftHandedPrice - normalPrice) * 100) / normalPrice

/-- Theorem stating the percentage increase in left-handed mouse price --/
theorem left_handed_mouse_price_increase 
  (store : StoreOperation)
  (h1 : store.daysOpen = 4)
  (h2 : store.miceSoldPerDay = 25)
  (h3 : store.normalMousePrice = 120)
  (h4 : store.weeklyRevenue = 15600) :
  percentageIncrease store.normalMousePrice 
    ((store.weeklyRevenue / store.daysOpen) / store.miceSoldPerDay) = 30 := by
  sorry

#eval percentageIncrease 120 156

end left_handed_mouse_price_increase_l3760_376097


namespace function_range_exclusion_l3760_376003

theorem function_range_exclusion (a : ℕ) : 
  (a > 3 → ∃ x : ℝ, -4 ≤ (8*x - 20) / (a - x^2) ∧ (8*x - 20) / (a - x^2) ≤ -1) ∧ 
  (∀ x : ℝ, (8*x - 20) / (3 - x^2) < -4 ∨ (8*x - 20) / (3 - x^2) > -1) :=
sorry

end function_range_exclusion_l3760_376003


namespace sqrt_four_squared_l3760_376012

theorem sqrt_four_squared : Real.sqrt (4^2) = 4 := by sorry

end sqrt_four_squared_l3760_376012


namespace coin_problem_l3760_376008

theorem coin_problem (x y z : ℕ) : 
  x + y + z = 30 →
  10 * x + 15 * y + 20 * z = 500 →
  z > x :=
by sorry

end coin_problem_l3760_376008


namespace candidate_admission_criterion_l3760_376034

/-- Represents the constructibility of an angle division -/
inductive AngleDivision
  | Constructible
  | NotConstructible

/-- Represents a candidate's response to the angle division questions -/
structure CandidateResponse :=
  (div19 : AngleDivision)
  (div17 : AngleDivision)
  (div18 : AngleDivision)

/-- Determines if an angle of n degrees can be divided into n equal parts -/
def canDivideAngle (n : ℕ) : AngleDivision :=
  if n = 19 ∨ n = 17 then AngleDivision.Constructible
  else AngleDivision.NotConstructible

/-- Determines if a candidate's response is correct -/
def isCorrectResponse (response : CandidateResponse) : Prop :=
  response.div19 = canDivideAngle 19 ∧
  response.div17 = canDivideAngle 17 ∧
  response.div18 = canDivideAngle 18

/-- Determines if a candidate should be admitted based on their response -/
def shouldAdmit (response : CandidateResponse) : Prop :=
  isCorrectResponse response

theorem candidate_admission_criterion (response : CandidateResponse) :
  response.div19 = AngleDivision.Constructible ∧
  response.div17 = AngleDivision.Constructible ∧
  response.div18 = AngleDivision.NotConstructible →
  shouldAdmit response :=
by sorry

end candidate_admission_criterion_l3760_376034


namespace elephants_after_three_years_is_zero_l3760_376020

/-- Represents the different types of animals in the zoo -/
inductive Animal
| Giraffe
| Penguin
| Elephant
| Lion
| Bear

/-- Represents the state of the zoo -/
structure ZooState where
  animalCount : Animal → ℕ
  budget : ℕ

/-- The cost of each animal type -/
def animalCost : Animal → ℕ
| Animal.Giraffe => 1000
| Animal.Penguin => 500
| Animal.Elephant => 1200
| Animal.Lion => 1100
| Animal.Bear => 1300

/-- The initial state of the zoo -/
def initialState : ZooState :=
  { animalCount := λ a => match a with
      | Animal.Giraffe => 5
      | Animal.Penguin => 10
      | Animal.Elephant => 0
      | Animal.Lion => 5
      | Animal.Bear => 0
    budget := 10000 }

/-- The maximum capacity of the zoo -/
def maxCapacity : ℕ := 300

/-- Theorem stating that the number of elephants after three years is zero -/
theorem elephants_after_three_years_is_zero :
  (initialState.animalCount Animal.Elephant) = 0 → 
  ∀ (finalState : ZooState),
    (finalState.animalCount Animal.Elephant) = 0 := by
  sorry

#check elephants_after_three_years_is_zero

end elephants_after_three_years_is_zero_l3760_376020


namespace four_bottles_cost_l3760_376026

/-- The cost of a certain number of bottles of mineral water -/
def cost (bottles : ℕ) : ℚ :=
  if bottles = 3 then 3/2 else (3/2 * bottles) / 3

/-- Theorem: The cost of 4 bottles of mineral water is 2 euros -/
theorem four_bottles_cost : cost 4 = 2 := by
  sorry

end four_bottles_cost_l3760_376026


namespace two_number_cards_totaling_twelve_probability_l3760_376011

/-- Represents a standard deck of cards -/
def StandardDeck : Type := Unit

/-- Number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- Set of card values that are numbers (2 through 10) -/
def numberCards : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9, 10}

/-- Number of cards of each value in the deck -/
def cardsPerValue : ℕ := 4

/-- Predicate for two cards totaling 12 -/
def totalTwelve (card1 card2 : ℕ) : Prop := card1 + card2 = 12

/-- The probability of the event -/
def probabilityTwoNumberCardsTotalingTwelve (deck : StandardDeck) : ℚ :=
  35 / 663

theorem two_number_cards_totaling_twelve_probability 
  (deck : StandardDeck) : 
  probabilityTwoNumberCardsTotalingTwelve deck = 35 / 663 := by
  sorry

end two_number_cards_totaling_twelve_probability_l3760_376011


namespace ellipse_chord_bisector_l3760_376022

def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

def point_inside_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 < 1

def bisector_line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

theorem ellipse_chord_bisector :
  ∀ x y : ℝ,
  ellipse x y →
  point_inside_ellipse 3 1 →
  bisector_line 3 4 (-13) x y :=
sorry

end ellipse_chord_bisector_l3760_376022


namespace line_up_count_l3760_376041

/-- The number of ways to arrange n distinct objects --/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange k distinct objects from n objects --/
def permutations (n k : ℕ) : ℕ := 
  if k > n then 0
  else factorial n / factorial (n - k)

/-- The number of boys in the group --/
def num_boys : ℕ := 2

/-- The number of girls in the group --/
def num_girls : ℕ := 3

/-- The total number of people in the group --/
def total_people : ℕ := num_boys + num_girls

theorem line_up_count : 
  factorial total_people - factorial (total_people - 1) * factorial num_boys = 72 := by
  sorry

end line_up_count_l3760_376041


namespace function_condition_implies_a_range_l3760_376081

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log (x + 1) - a * x

theorem function_condition_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → (f a x + a * x) / exp x ≤ a * x) ↔ a ≥ 1 :=
by sorry

end function_condition_implies_a_range_l3760_376081


namespace taxi_journey_theorem_l3760_376001

def itinerary : List Int := [-15, 4, -5, 10, -12, 5, 8, -7]
def gasoline_consumption : Rat := 10 / 100  -- 10 liters per 100 km
def gasoline_price : Rat := 8  -- 8 yuan per liter

def total_distance (route : List Int) : Int :=
  route.map (Int.natAbs) |>.sum

theorem taxi_journey_theorem :
  let distance := total_distance itinerary
  let cost := (distance : Rat) * gasoline_consumption * gasoline_price
  distance = 66 ∧ cost = 52.8 := by sorry

end taxi_journey_theorem_l3760_376001


namespace arithmetic_sequence_common_difference_l3760_376079

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  sum : ℕ → ℚ -- Sum function
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2
  term_formula : ∀ n, a n = a 1 + (n - 1) * d

/-- The common difference of the arithmetic sequence is 4 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence)
  (sum_5 : seq.sum 5 = -15)
  (sum_terms : seq.a 2 + seq.a 5 = -2) :
  seq.d = 4 := by
  sorry

end arithmetic_sequence_common_difference_l3760_376079


namespace initial_minutes_plan_a_l3760_376075

/-- Represents the cost in dollars for a call under Plan A -/
def costPlanA (initialMinutes : ℕ) (totalMinutes : ℕ) : ℚ :=
  0.60 + 0.06 * (totalMinutes - initialMinutes)

/-- Represents the cost in dollars for a call under Plan B -/
def costPlanB (minutes : ℕ) : ℚ :=
  0.08 * minutes

theorem initial_minutes_plan_a : ∃ (x : ℕ), 
  (∀ (m : ℕ), m ≥ x → costPlanA x m = costPlanB m) ∧
  (costPlanA x 18 = costPlanB 18) ∧
  x = 4 := by
  sorry

end initial_minutes_plan_a_l3760_376075


namespace some_trinks_not_zorbs_l3760_376028

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for Zorb, Glarb, and Trink
variable (Zorb Glarb Trink : U → Prop)

-- Hypothesis I: All Zorbs are not Glarbs
variable (h1 : ∀ x, Zorb x → ¬Glarb x)

-- Hypothesis II: Some Glarbs are Trinks
variable (h2 : ∃ x, Glarb x ∧ Trink x)

-- Theorem: Some Trinks are not Zorbs
theorem some_trinks_not_zorbs :
  ∃ x, Trink x ∧ ¬Zorb x :=
sorry

end some_trinks_not_zorbs_l3760_376028


namespace filter_price_theorem_l3760_376002

-- Define the number of filters and their prices
def total_filters : ℕ := 5
def kit_price : ℚ := 87.50
def price_filter_1 : ℚ := 16.45
def price_filter_2 : ℚ := 19.50
def num_filter_1 : ℕ := 2
def num_filter_2 : ℕ := 1
def num_unknown_price : ℕ := 2
def savings_percentage : ℚ := 0.08

-- Define the function to calculate the total individual price
def total_individual_price (x : ℚ) : ℚ :=
  num_filter_1 * price_filter_1 + num_unknown_price * x + num_filter_2 * price_filter_2

-- Define the theorem
theorem filter_price_theorem (x : ℚ) :
  (savings_percentage * total_individual_price x = total_individual_price x - kit_price) →
  x = 21.36 := by
  sorry

end filter_price_theorem_l3760_376002


namespace average_bull_weight_l3760_376037

/-- Represents a section of the farm with a ratio of cows to bulls -/
structure FarmSection where
  cows : ℕ
  bulls : ℕ

/-- Represents the farm with its sections and total cattle -/
structure Farm where
  sectionA : FarmSection
  sectionB : FarmSection
  sectionC : FarmSection
  totalCattle : ℕ
  totalBullWeight : ℕ

def farm : Farm := {
  sectionA := { cows := 7, bulls := 21 },
  sectionB := { cows := 5, bulls := 15 },
  sectionC := { cows := 3, bulls := 9 },
  totalCattle := 1220,
  totalBullWeight := 200000
}

theorem average_bull_weight (f : Farm) :
  f = farm →
  (f.totalBullWeight : ℚ) / (((f.sectionA.bulls + f.sectionB.bulls + f.sectionC.bulls) * f.totalCattle) / (f.sectionA.cows + f.sectionA.bulls + f.sectionB.cows + f.sectionB.bulls + f.sectionC.cows + f.sectionC.bulls)) = 218579 / 1000 := by
  sorry

end average_bull_weight_l3760_376037


namespace garret_age_proof_l3760_376031

/-- Garret's current age -/
def garret_age : ℕ := 12

/-- Shane's current age -/
def shane_current_age : ℕ := 44

theorem garret_age_proof :
  (shane_current_age - 20 = 2 * garret_age) →
  garret_age = 12 := by
sorry

end garret_age_proof_l3760_376031


namespace weighted_average_theorem_l3760_376021

def group1_avg : ℝ := 30
def group1_weight : ℝ := 2
def group2_avg : ℝ := 40
def group2_weight : ℝ := 3
def group3_avg : ℝ := 20
def group3_weight : ℝ := 1

def total_weighted_sum : ℝ := group1_avg * group1_weight + group2_avg * group2_weight + group3_avg * group3_weight
def total_weight : ℝ := group1_weight + group2_weight + group3_weight

theorem weighted_average_theorem : total_weighted_sum / total_weight = 200 / 6 := by
  sorry

end weighted_average_theorem_l3760_376021


namespace right_triangle_angles_l3760_376087

/-- A right-angled triangle with a specific property -/
structure RightTriangle where
  /-- The measure of the right angle in degrees -/
  right_angle : ℝ
  /-- The measure of the angle between the angle bisector of the right angle and the median to the hypotenuse, in degrees -/
  bisector_median_angle : ℝ
  /-- The right angle is 90 degrees -/
  right_angle_is_90 : right_angle = 90
  /-- The angle between the bisector and median is 16 degrees -/
  bisector_median_angle_is_16 : bisector_median_angle = 16

/-- The angles of the triangle given the specific conditions -/
def triangle_angles (t : RightTriangle) : (ℝ × ℝ × ℝ) :=
  (61, 29, 90)

/-- Theorem stating that the angles of the triangle are 61°, 29°, and 90° given the conditions -/
theorem right_triangle_angles (t : RightTriangle) :
  triangle_angles t = (61, 29, 90) := by
  sorry

end right_triangle_angles_l3760_376087


namespace olympiad_colors_l3760_376019

-- Define the colors
inductive Color
  | Red
  | Yellow
  | Blue

-- Define a person's outfit
structure Outfit :=
  (dress : Color)
  (notebook : Color)

-- Define the problem statement
theorem olympiad_colors :
  ∃ (sveta tanya ira : Outfit),
    -- All dress colors are different
    sveta.dress ≠ tanya.dress ∧ sveta.dress ≠ ira.dress ∧ tanya.dress ≠ ira.dress ∧
    -- All notebook colors are different
    sveta.notebook ≠ tanya.notebook ∧ sveta.notebook ≠ ira.notebook ∧ tanya.notebook ≠ ira.notebook ∧
    -- Only Sveta's dress and notebook colors match
    (sveta.dress = sveta.notebook) ∧
    (tanya.dress ≠ tanya.notebook) ∧
    (ira.dress ≠ ira.notebook) ∧
    -- Tanya's dress and notebook are not red
    (tanya.dress ≠ Color.Red) ∧ (tanya.notebook ≠ Color.Red) ∧
    -- Ira has a yellow notebook
    (ira.notebook = Color.Yellow) ∧
    -- The solution
    sveta = Outfit.mk Color.Red Color.Red ∧
    ira = Outfit.mk Color.Blue Color.Yellow ∧
    tanya = Outfit.mk Color.Yellow Color.Blue :=
by
  sorry

end olympiad_colors_l3760_376019


namespace sculpture_cost_in_inr_l3760_376010

/-- Exchange rate between US dollars and Namibian dollars -/
def usd_to_nad : ℝ := 10

/-- Exchange rate between US dollars and Chinese yuan -/
def usd_to_cny : ℝ := 7

/-- Exchange rate between Chinese yuan and Indian Rupees -/
def cny_to_inr : ℝ := 10

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 200

/-- Theorem stating the cost of the sculpture in Indian Rupees -/
theorem sculpture_cost_in_inr :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny * cny_to_inr = 1400 := by
  sorry

end sculpture_cost_in_inr_l3760_376010


namespace option_b_is_best_l3760_376066

-- Define the problem parameters
def total_metal_needed : ℝ := 635
def metal_in_storage : ℝ := 276
def aluminum_percentage : ℝ := 0.60
def steel_percentage : ℝ := 0.40

-- Define supplier options
structure Supplier :=
  (aluminum_price : ℝ)
  (steel_price : ℝ)

def option_a : Supplier := ⟨1.30, 0.90⟩
def option_b : Supplier := ⟨1.10, 1.00⟩
def option_c : Supplier := ⟨1.25, 0.95⟩

-- Calculate additional metal needed
def additional_metal_needed : ℝ := total_metal_needed - metal_in_storage

-- Calculate cost for a supplier
def calculate_cost (s : Supplier) : ℝ :=
  (additional_metal_needed * aluminum_percentage * s.aluminum_price) +
  (additional_metal_needed * steel_percentage * s.steel_price)

-- Theorem to prove
theorem option_b_is_best :
  calculate_cost option_b < calculate_cost option_a ∧
  calculate_cost option_b < calculate_cost option_c :=
by sorry

end option_b_is_best_l3760_376066


namespace tangent_line_and_monotonicity_and_range_l3760_376072

noncomputable section

open Real

-- Define f(x) = ln x
def f (x : ℝ) : ℝ := log x

-- Define g(x) = f(x) + f''(x)
def g (x : ℝ) : ℝ := f x + (deriv^[2] f) x

theorem tangent_line_and_monotonicity_and_range :
  -- 1. The tangent line to y = f(x) at (1, f(1)) is y = x - 1
  (∀ y, y = deriv f 1 * (x - 1) + f 1 ↔ y = x - 1) ∧
  -- 2. g(x) is decreasing on (0, 1) and increasing on (1, +∞)
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → g x₁ > g x₂) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → g x₁ < g x₂) ∧
  -- 3. For any x > 0, g(a) - g(x) < 1/a holds if and only if 0 < a < e
  (∀ a, (0 < a ∧ a < ℯ) ↔ (∀ x, x > 0 → g a - g x < 1 / a)) :=
sorry

end

end tangent_line_and_monotonicity_and_range_l3760_376072


namespace unique_three_digit_number_l3760_376083

theorem unique_three_digit_number : ∃! n : ℕ,
  (100 ≤ n ∧ n ≤ 999) ∧
  (n % 11 = 0) ∧
  (n / 11 = (n / 100)^2 + ((n / 10) % 10)^2 + (n % 10)^2) ∧
  (n = 550) := by
sorry

end unique_three_digit_number_l3760_376083


namespace total_bars_is_504_l3760_376061

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 18

/-- The number of chocolate bars in each small box -/
def bars_per_small_box : ℕ := 28

/-- The total number of chocolate bars in the large box -/
def total_chocolate_bars : ℕ := num_small_boxes * bars_per_small_box

/-- Theorem: The total number of chocolate bars in the large box is 504 -/
theorem total_bars_is_504 : total_chocolate_bars = 504 := by
  sorry

end total_bars_is_504_l3760_376061


namespace prime_factors_count_l3760_376092

/-- The total number of prime factors in the expression (4)^11 × (7)^3 × (11)^2 -/
def totalPrimeFactors : ℕ := 27

/-- The exponent of 4 in the expression -/
def exponent4 : ℕ := 11

/-- The exponent of 7 in the expression -/
def exponent7 : ℕ := 3

/-- The exponent of 11 in the expression -/
def exponent11 : ℕ := 2

theorem prime_factors_count : 
  totalPrimeFactors = 2 * exponent4 + exponent7 + exponent11 := by
  sorry

end prime_factors_count_l3760_376092


namespace trail_distribution_count_l3760_376054

/-- The number of ways to distribute 4 people familiar with trails into two groups of 2 each -/
def trail_distribution_ways : ℕ := Nat.choose 4 2

/-- Theorem stating that the number of ways to distribute 4 people familiar with trails
    into two groups of 2 each is equal to 6 -/
theorem trail_distribution_count : trail_distribution_ways = 6 := by
  sorry

end trail_distribution_count_l3760_376054


namespace sotka_not_divisible_by_nine_l3760_376030

/-- Represents a digit in the range 0 to 9 -/
def Digit := Fin 10

/-- Represents the mapping of letters to digits -/
def LetterToDigit := Char → Digit

/-- Checks if all characters in a string are mapped to unique digits -/
def allUnique (s : String) (m : LetterToDigit) : Prop :=
  ∀ c₁ c₂, c₁ ∈ s.data → c₂ ∈ s.data → c₁ ≠ c₂ → m c₁ ≠ m c₂

/-- Converts a string to a number using the given mapping -/
def toNumber (s : String) (m : LetterToDigit) : ℕ :=
  s.data.foldr (λ c acc => acc * 10 + (m c).val) 0

/-- The main theorem -/
theorem sotka_not_divisible_by_nine (m : LetterToDigit) : 
  allUnique "ДЕВЯНОСТО" m →
  allUnique "ДЕВЯТКА" m →
  allUnique "СОТКА" m →
  90 ∣ toNumber "ДЕВЯНОСТО" m →
  9 ∣ toNumber "ДЕВЯТКА" m →
  ¬(9 ∣ toNumber "СОТКА" m) := by
  sorry


end sotka_not_divisible_by_nine_l3760_376030


namespace trisha_works_52_weeks_l3760_376049

/-- Calculates the number of weeks worked in a year based on given parameters -/
def weeks_worked (hourly_rate : ℚ) (hours_per_week : ℚ) (withholding_rate : ℚ) (annual_take_home : ℚ) : ℚ :=
  annual_take_home / ((hourly_rate * hours_per_week) * (1 - withholding_rate))

/-- Proves that given the specified parameters, Trisha works 52 weeks in a year -/
theorem trisha_works_52_weeks :
  weeks_worked 15 40 (1/5) 24960 = 52 := by
  sorry

end trisha_works_52_weeks_l3760_376049


namespace sum_mod_nine_l3760_376042

theorem sum_mod_nine : (9156 + 9157 + 9158 + 9159 + 9160) % 9 = 7 := by
  sorry

end sum_mod_nine_l3760_376042


namespace total_trip_cost_l3760_376089

def rental_cost : ℝ := 150
def gas_price : ℝ := 3.50
def gas_purchased : ℝ := 8
def mileage_cost : ℝ := 0.50
def distance_driven : ℝ := 320

theorem total_trip_cost : 
  rental_cost + gas_price * gas_purchased + mileage_cost * distance_driven = 338 := by
  sorry

end total_trip_cost_l3760_376089


namespace infinite_solutions_and_sum_of_exceptions_l3760_376073

/-- Given an equation (x+B)(Ax+40) / ((x+C)(x+8)) = 3, this theorem proves that
    for specific values of A, B, and C, the equation has infinitely many solutions,
    and provides the sum of x values that do not satisfy the equation. -/
theorem infinite_solutions_and_sum_of_exceptions :
  ∃ (A B C : ℚ),
    (A = 3 ∧ B = 8 ∧ C = 40/3) ∧
    (∀ x : ℚ, x ≠ -C → x ≠ -8 →
      (x + B) * (A * x + 40) / ((x + C) * (x + 8)) = 3) ∧
    ((-8) + (-40/3) = -64/3) := by
  sorry


end infinite_solutions_and_sum_of_exceptions_l3760_376073


namespace women_workers_l3760_376098

/-- Represents a company with workers and retirement plans. -/
structure Company where
  total_workers : ℕ
  workers_without_plan : ℕ
  women_without_plan : ℕ
  men_with_plan : ℕ
  total_men : ℕ

/-- Conditions for the company structure -/
def company_conditions (c : Company) : Prop :=
  c.workers_without_plan = c.total_workers / 3 ∧
  c.women_without_plan = (2 * c.workers_without_plan) / 5 ∧
  c.men_with_plan = ((2 * c.total_workers) / 3) * 2 / 5 ∧
  c.total_men = 120

/-- The theorem to prove -/
theorem women_workers (c : Company) 
  (h : company_conditions c) : c.total_workers - c.total_men = 330 := by
  sorry

#check women_workers

end women_workers_l3760_376098


namespace three_student_committees_l3760_376062

theorem three_student_committees (n k : ℕ) (hn : n = 10) (hk : k = 3) :
  Nat.choose n k = 120 := by
  sorry

end three_student_committees_l3760_376062


namespace smallest_divisible_by_2000_l3760_376032

def sequence_a (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (n - 1 : ℤ) * a (n + 1) = (n + 1 : ℤ) * a n - 2 * (n - 1 : ℤ)

theorem smallest_divisible_by_2000 (a : ℕ → ℤ) (h : sequence_a a) (h2000 : 2000 ∣ a 1999) :
  (∃ n : ℕ, n ≥ 2 ∧ 2000 ∣ a n) ∧ (∀ m : ℕ, m ≥ 2 ∧ m < 249 → ¬(2000 ∣ a m)) ∧ 2000 ∣ a 249 :=
by sorry

end smallest_divisible_by_2000_l3760_376032
