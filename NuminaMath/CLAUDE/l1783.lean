import Mathlib

namespace max_fourth_power_sum_l1783_178309

theorem max_fourth_power_sum (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 8) :
  ∃ (m : ℝ), m = 16 ∧ a^4 + b^4 + c^4 + d^4 ≤ m ∧
  ∃ (a' b' c' d' : ℝ), a'^3 + b'^3 + c'^3 + d'^3 = 8 ∧ a'^4 + b'^4 + c'^4 + d'^4 = m :=
by sorry

end max_fourth_power_sum_l1783_178309


namespace hamburger_count_l1783_178308

theorem hamburger_count (total_spent single_cost double_cost double_count : ℚ) 
  (h1 : total_spent = 64.5)
  (h2 : single_cost = 1)
  (h3 : double_cost = 1.5)
  (h4 : double_count = 29) :
  ∃ (single_count : ℚ), 
    single_count * single_cost + double_count * double_cost = total_spent ∧ 
    single_count + double_count = 50 := by
  sorry

end hamburger_count_l1783_178308


namespace original_ratio_proof_l1783_178387

theorem original_ratio_proof (initial_boarders : ℕ) (new_boarders : ℕ) :
  initial_boarders = 220 →
  new_boarders = 44 →
  (initial_boarders + new_boarders) * 2 = (initial_boarders + new_boarders + (initial_boarders + new_boarders) * 2) →
  (5 : ℚ) / 12 = initial_boarders / ((initial_boarders + new_boarders) * 2 : ℚ) :=
by
  sorry

end original_ratio_proof_l1783_178387


namespace cube_equation_solution_l1783_178337

theorem cube_equation_solution :
  ∃ x : ℝ, (x - 5)^3 = (1/27)⁻¹ ∧ x = 8 :=
by sorry

end cube_equation_solution_l1783_178337


namespace f_iterative_application_l1783_178365

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 - 3*x + 2 else x + 10

theorem f_iterative_application : f (f (f (f (f 2)))) = -8 := by
  sorry

end f_iterative_application_l1783_178365


namespace bubble_theorem_l1783_178345

/-- The number of bubbles appearing each minute -/
def k : ℕ := 36

/-- The number of minutes after which bubbles start bursting -/
def m : ℕ := 80

/-- The maximum number of bubbles on the screen -/
def max_bubbles : ℕ := k * (k + 21) / 2

theorem bubble_theorem :
  (∀ n : ℕ, n ≤ 10 + m → n * k = n * k) ∧  -- Bubbles appear every minute
  ((10 + m) * k = m * (m + 1) / 2) ∧  -- All bubbles eventually burst
  (∀ n : ℕ, n ≤ m → n * (n + 1) / 2 ≤ (10 + n) * k) ∧  -- Bursting pattern
  (k * (k + 21) / 2 = 1026) →  -- Definition of max_bubbles
  max_bubbles = 1026 := by sorry

#eval max_bubbles  -- Should output 1026

end bubble_theorem_l1783_178345


namespace longest_segment_in_cylinder_l1783_178317

/-- The longest segment in a cylinder with radius 5 and height 10 is 10√2 -/
theorem longest_segment_in_cylinder : ∀ (r h : ℝ),
  r = 5 → h = 10 → 
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 10 * Real.sqrt 2 :=
by sorry

end longest_segment_in_cylinder_l1783_178317


namespace not_divisible_1985_1987_divisible_1987_1989_l1783_178343

/-- Represents an L-shaped piece consisting of 3 unit squares -/
structure LShape :=
  (width : ℕ)
  (height : ℕ)
  (area_eq_3 : width * height = 3)

/-- Checks if a rectangle can be divided into L-shapes -/
def can_divide_into_l_shapes (m n : ℕ) : Prop :=
  (m * n) % 3 = 0 ∨ 
  ∃ (a b c d : ℕ), m = 2 * a + 7 * b ∧ n = 3 * c + 9 * d

/-- Theorem stating the divisibility condition for 1985 × 1987 rectangle -/
theorem not_divisible_1985_1987 : ¬(can_divide_into_l_shapes 1985 1987) :=
sorry

/-- Theorem stating the divisibility condition for 1987 × 1989 rectangle -/
theorem divisible_1987_1989 : can_divide_into_l_shapes 1987 1989 :=
sorry

end not_divisible_1985_1987_divisible_1987_1989_l1783_178343


namespace surface_classification_l1783_178352

/-- A surface in 3D space -/
inductive Surface
  | CircularCone
  | OneSheetHyperboloid
  | TwoSheetHyperboloid
  | EllipticParaboloid

/-- Determine the type of surface given its equation -/
def determine_surface_type (equation : ℝ → ℝ → ℝ → Prop) : Surface :=
  sorry

theorem surface_classification :
  (determine_surface_type (fun x y z => x^2 - y^2 = z^2) = Surface.CircularCone) ∧
  (determine_surface_type (fun x y z => -2*x^2 + 2*y^2 + z^2 = 4) = Surface.OneSheetHyperboloid) ∧
  (determine_surface_type (fun x y z => 2*x^2 - y^2 + z^2 + 2 = 0) = Surface.TwoSheetHyperboloid) ∧
  (determine_surface_type (fun x y z => 3*y^2 + 2*z^2 = 6*x) = Surface.EllipticParaboloid) :=
by
  sorry

end surface_classification_l1783_178352


namespace base7_multiplication_l1783_178356

/-- Converts a number from base 7 to base 10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Represents a number in base 7 --/
structure Base7 where
  value : ℕ

theorem base7_multiplication :
  let a : Base7 := ⟨345⟩
  let b : Base7 := ⟨3⟩
  let result : Base7 := ⟨1401⟩
  toBase7 (toBase10 a.value * toBase10 b.value) = result.value := by sorry

end base7_multiplication_l1783_178356


namespace urn_probability_l1783_178368

/-- Represents the content of the urn -/
structure UrnContent where
  red : ℕ
  blue : ℕ

/-- Represents a single draw operation -/
inductive DrawResult
| Red
| Blue

/-- Represents a sequence of draw results -/
def DrawSequence := List DrawResult

/-- The initial content of the urn -/
def initial_urn : UrnContent := ⟨2, 1⟩

/-- The number of draw operations performed -/
def num_operations : ℕ := 5

/-- The final number of balls in the urn -/
def final_total_balls : ℕ := 8

/-- The target final content of the urn -/
def target_final_urn : UrnContent := ⟨3, 5⟩

/-- Calculates the probability of drawing a red ball from the urn -/
def prob_draw_red (urn : UrnContent) : ℚ :=
  urn.red / (urn.red + urn.blue)

/-- Updates the urn content after a draw -/
def update_urn (urn : UrnContent) (draw : DrawResult) : UrnContent :=
  match draw with
  | DrawResult.Red => ⟨urn.red + 1, urn.blue⟩
  | DrawResult.Blue => ⟨urn.red, urn.blue + 1⟩

/-- Calculates the probability of a specific draw sequence -/
def sequence_probability (seq : DrawSequence) : ℚ :=
  sorry

/-- Calculates the number of valid sequences leading to the target urn content -/
def num_valid_sequences : ℕ :=
  sorry

/-- The main theorem stating the probability of ending with the target urn content -/
theorem urn_probability : 
  sequence_probability (List.replicate num_operations DrawResult.Red) * num_valid_sequences = 4/21 :=
sorry

end urn_probability_l1783_178368


namespace carpet_area_l1783_178321

theorem carpet_area : 
  ∀ (length width : ℝ) (shoe_length : ℝ),
    shoe_length = 28 →
    length = 15 * shoe_length →
    width = 10 * shoe_length →
    length * width = 117600 := by
  sorry

end carpet_area_l1783_178321


namespace largest_five_digit_congruent_17_mod_26_l1783_178378

theorem largest_five_digit_congruent_17_mod_26 : ∃ (n : ℕ), n = 99997 ∧ 
  n < 100000 ∧ 
  n % 26 = 17 ∧ 
  ∀ (m : ℕ), m < 100000 → m % 26 = 17 → m ≤ n :=
by sorry

end largest_five_digit_congruent_17_mod_26_l1783_178378


namespace white_beans_count_l1783_178335

/-- The number of white jelly beans in one bag -/
def white_beans_in_bag : ℕ := sorry

/-- The number of bags needed to fill the fishbowl -/
def bags_in_fishbowl : ℕ := 3

/-- The number of red jelly beans in one bag -/
def red_beans_in_bag : ℕ := 24

/-- The total number of red and white jelly beans in the fishbowl -/
def total_red_white_in_fishbowl : ℕ := 126

theorem white_beans_count : white_beans_in_bag = 18 := by
  sorry

end white_beans_count_l1783_178335


namespace imaginary_part_of_minus_one_plus_i_squared_l1783_178350

theorem imaginary_part_of_minus_one_plus_i_squared :
  Complex.im ((-1 + Complex.I) ^ 2) = -2 := by
sorry

end imaginary_part_of_minus_one_plus_i_squared_l1783_178350


namespace equation_solution_l1783_178374

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (3*x + 7)*(x - 2) - (7*x - 4)
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 39 / 3 ∧
              x₂ = 1 - Real.sqrt 39 / 3 ∧
              f x₁ = 0 ∧
              f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end equation_solution_l1783_178374


namespace cube_root_over_sixth_root_of_eight_l1783_178377

theorem cube_root_over_sixth_root_of_eight (x : ℝ) :
  (8 ^ (1/3)) / (8 ^ (1/6)) = 8 ^ (1/6) :=
by sorry

end cube_root_over_sixth_root_of_eight_l1783_178377


namespace wire_service_reporters_l1783_178346

theorem wire_service_reporters (x y both_local non_local_politics international_only : ℝ) 
  (hx : x = 35)
  (hy : y = 25)
  (hboth : both_local = 20)
  (hnon_local : non_local_politics = 30)
  (hinter : international_only = 15) :
  100 - ((x + y - both_local) + non_local_politics + international_only) = 75 := by
  sorry

end wire_service_reporters_l1783_178346


namespace range_of_a_for_quadratic_inequality_l1783_178382

theorem range_of_a_for_quadratic_inequality :
  (∀ x : ℝ, ∀ a : ℝ, a * x^2 - a * x - 1 ≤ 0) →
  (∀ a : ℝ, (a ∈ Set.Icc (-4 : ℝ) 0) ↔ (∀ x : ℝ, a * x^2 - a * x - 1 ≤ 0)) :=
by sorry

end range_of_a_for_quadratic_inequality_l1783_178382


namespace min_k_for_sqrt_inequality_l1783_178329

theorem min_k_for_sqrt_inequality : 
  ∃ k : ℝ, k = Real.sqrt 2 ∧ 
  (∀ x y : ℝ, Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (x + y)) ∧
  (∀ k' : ℝ, k' < k → 
    ∃ x y : ℝ, Real.sqrt x + Real.sqrt y > k' * Real.sqrt (x + y)) := by
  sorry

end min_k_for_sqrt_inequality_l1783_178329


namespace smallest_factorization_coefficient_l1783_178310

theorem smallest_factorization_coefficient (b : ℕ+) : 
  (∃ (r s : ℤ), (∀ x : ℝ, x^2 + b.val*x + 3258 = (x + r) * (x + s))) →
  b.val ≥ 1089 :=
sorry

end smallest_factorization_coefficient_l1783_178310


namespace search_plans_count_l1783_178385

/-- Represents the number of children in the group -/
def total_children : ℕ := 6

/-- Represents the number of food drop locations -/
def num_locations : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of search plans when Grace doesn't participate -/
def plans_without_grace : ℕ := 
  (choose (total_children - 1) 1) * ((choose (total_children - 2) 2) / 2) * (Nat.factorial num_locations)

/-- Calculates the number of search plans when Grace participates -/
def plans_with_grace : ℕ := choose (total_children - 1) 2

/-- The total number of different search plans -/
def total_plans : ℕ := plans_without_grace + plans_with_grace

/-- Theorem stating that the total number of different search plans is 40 -/
theorem search_plans_count : total_plans = 40 := by
  sorry


end search_plans_count_l1783_178385


namespace mode_of_sports_shoes_l1783_178318

/-- Represents the sales data for a particular shoe size -/
structure SalesData :=
  (size : Float)
  (sales : Nat)

/-- Finds the mode of a list of SalesData -/
def findMode (data : List SalesData) : Float :=
  sorry

/-- The sales data for the sports shoes -/
def salesData : List SalesData := [
  ⟨24, 1⟩,
  ⟨24.5, 3⟩,
  ⟨25, 10⟩,
  ⟨25.5, 4⟩,
  ⟨26, 2⟩
]

theorem mode_of_sports_shoes :
  findMode salesData = 25 := by
  sorry

end mode_of_sports_shoes_l1783_178318


namespace rectangle_horizontal_length_l1783_178344

/-- Proves that a rectangle with perimeter 54 cm and horizontal length 3 cm longer than vertical length has a horizontal length of 15 cm -/
theorem rectangle_horizontal_length : 
  ∀ (v h : ℝ), 
  (2 * v + 2 * h = 54) →  -- perimeter is 54 cm
  (h = v + 3) →           -- horizontal length is 3 cm longer than vertical length
  h = 15 := by            -- horizontal length is 15 cm
sorry

end rectangle_horizontal_length_l1783_178344


namespace decagon_triangles_l1783_178389

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def num_triangles_in_decagon : ℕ := 120

/-- Theorem: The number of triangles that can be formed using the vertices of a regular decagon is 120 -/
theorem decagon_triangles : num_triangles_in_decagon = 120 := by
  sorry

end decagon_triangles_l1783_178389


namespace sum_of_a_and_b_l1783_178362

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 47) 
  (eq2 : 4 * a + 3 * b = 39) : 
  a + b = 82 / 7 := by
  sorry

end sum_of_a_and_b_l1783_178362


namespace min_sequence_length_l1783_178323

def S : Finset Nat := {1, 2, 3, 4}

def is_valid_sequence (a : List Nat) : Prop :=
  ∀ b : List Nat, b.length = 4 ∧ b.toFinset = S ∧ b.getLast? ≠ some 1 →
    ∃ i₁ i₂ i₃ i₄, i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧ i₄ ≤ a.length ∧
      (a.get? i₁, a.get? i₂, a.get? i₃, a.get? i₄) = (b.get? 0, b.get? 1, b.get? 2, b.get? 3)

theorem min_sequence_length :
  ∃ a : List Nat, a.length = 11 ∧ is_valid_sequence a ∧
    ∀ a' : List Nat, is_valid_sequence a' → a'.length ≥ 11 := by
  sorry

end min_sequence_length_l1783_178323


namespace systematic_sample_valid_l1783_178373

def is_valid_systematic_sample (sample : List Nat) (population_size : Nat) (sample_size : Nat) : Prop :=
  sample.length = sample_size ∧
  ∀ i j, i < j → i < sample.length → j < sample.length →
    sample[i]! < sample[j]! ∧
    (sample[j]! - sample[i]!) % (population_size / sample_size) = 0 ∧
    sample[sample.length - 1]! ≤ population_size

theorem systematic_sample_valid :
  is_valid_systematic_sample [3, 13, 23, 33, 43, 53] 60 6 := by
  sorry

end systematic_sample_valid_l1783_178373


namespace trig_identity_l1783_178340

theorem trig_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := by
  sorry

end trig_identity_l1783_178340


namespace evaluate_expression_l1783_178314

theorem evaluate_expression (a b : ℚ) (ha : a = 3) (hb : b = 2) :
  (a^4 + b^4) / (a^2 - a*b + b^2) = 97 / 7 := by
  sorry

end evaluate_expression_l1783_178314


namespace fred_total_games_l1783_178397

/-- The total number of basketball games Fred attended over two years -/
def total_games (games_this_year games_last_year : ℕ) : ℕ :=
  games_this_year + games_last_year

/-- Theorem stating that Fred attended 85 games in total -/
theorem fred_total_games : 
  total_games 60 25 = 85 := by
  sorry

end fred_total_games_l1783_178397


namespace fraction_equality_l1783_178354

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = -4 / 3 := by
sorry

end fraction_equality_l1783_178354


namespace root_conditions_imply_m_range_l1783_178364

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 3

-- State the theorem
theorem root_conditions_imply_m_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0 ∧ x > 1 ∧ y < 1) →
  m > 4 :=
sorry

end root_conditions_imply_m_range_l1783_178364


namespace basketball_probability_l1783_178320

theorem basketball_probability (jack_prob jill_prob sandy_prob : ℚ) 
  (h1 : jack_prob = 1/6)
  (h2 : jill_prob = 1/7)
  (h3 : sandy_prob = 1/8) :
  (1 - jack_prob) * jill_prob * sandy_prob = 5/336 := by
  sorry

end basketball_probability_l1783_178320


namespace valleyball_league_members_l1783_178393

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 5

/-- The total cost for all members in dollars -/
def total_cost : ℕ := 3300

/-- The number of pairs of socks each member needs -/
def socks_per_member : ℕ := 2

/-- The number of T-shirts each member needs -/
def tshirts_per_member : ℕ := 2

/-- The number of members in the Valleyball Soccer League -/
def number_of_members : ℕ := 97

theorem valleyball_league_members :
  let tshirt_cost := sock_cost + tshirt_additional_cost
  let member_cost := socks_per_member * sock_cost + tshirts_per_member * tshirt_cost
  number_of_members * member_cost = total_cost := by
  sorry


end valleyball_league_members_l1783_178393


namespace range_of_a_l1783_178327

theorem range_of_a (a : ℝ) 
  (p : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x)
  (q : ∃ x₀ : ℝ, x₀^2 + 4*x₀ + a = 0) :
  e ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l1783_178327


namespace difference_of_squares_64_36_l1783_178300

theorem difference_of_squares_64_36 : 64^2 - 36^2 = 2800 := by
  sorry

end difference_of_squares_64_36_l1783_178300


namespace fourth_power_equation_l1783_178353

theorem fourth_power_equation : 10^4 + 15^4 + 8^4 + 2*3^4 = 16^4 := by
  sorry

end fourth_power_equation_l1783_178353


namespace total_weight_of_balls_l1783_178336

def blue_ball_weight : ℝ := 6
def brown_ball_weight : ℝ := 3.12

theorem total_weight_of_balls :
  blue_ball_weight + brown_ball_weight = 9.12 := by
  sorry

end total_weight_of_balls_l1783_178336


namespace sample_capacity_proof_l1783_178322

theorem sample_capacity_proof (n : ℕ) (frequency : ℕ) (relative_frequency : ℚ) 
  (h1 : frequency = 30)
  (h2 : relative_frequency = 1/4)
  (h3 : relative_frequency = frequency / n) :
  n = 120 := by
  sorry

end sample_capacity_proof_l1783_178322


namespace right_triangle_perimeter_l1783_178341

theorem right_triangle_perimeter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_area : (1/2) * a * b = 150) (h_leg : a = 30) : 
  a + b + c = 40 + 10 * Real.sqrt 10 := by
  sorry

end right_triangle_perimeter_l1783_178341


namespace car_distance_theorem_l1783_178301

/-- Given a car traveling at a constant speed for a certain time, 
    calculate the distance covered. -/
def distance_covered (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem stating that a car traveling at 107 km/h for 6.5 hours
    covers a distance of 695.5 km. -/
theorem car_distance_theorem :
  distance_covered 107 6.5 = 695.5 := by
  sorry

end car_distance_theorem_l1783_178301


namespace kelly_games_theorem_l1783_178355

/-- The number of games Kelly needs to give away to reach her desired number of games -/
def games_to_give_away (initial_games desired_games : ℕ) : ℕ :=
  initial_games - desired_games

theorem kelly_games_theorem (initial_games desired_games : ℕ) 
  (h1 : initial_games = 120) (h2 : desired_games = 20) : 
  games_to_give_away initial_games desired_games = 100 := by
  sorry

end kelly_games_theorem_l1783_178355


namespace arithmetic_sequence_terms_l1783_178315

theorem arithmetic_sequence_terms (a₁ : ℕ) (d : ℤ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 20 ∧ d = -2 ∧ aₙ = 10 ∧ aₙ = a₁ + (n - 1) * d → n = 6 :=
by sorry

end arithmetic_sequence_terms_l1783_178315


namespace amc_12_scoring_problem_l1783_178399

/-- The minimum number of correctly solved problems to achieve the target score -/
def min_correct_problems (total_problems : ℕ) (attempted_problems : ℕ) (points_correct : ℕ) 
  (points_unanswered : ℕ) (target_score : ℕ) : ℕ :=
  let unanswered := total_problems - attempted_problems
  let points_from_unanswered := unanswered * points_unanswered
  let required_points := target_score - points_from_unanswered
  (required_points + points_correct - 1) / points_correct

theorem amc_12_scoring_problem :
  min_correct_problems 30 25 7 2 120 = 16 := by
  sorry

end amc_12_scoring_problem_l1783_178399


namespace emily_fish_weight_l1783_178302

/-- Calculates the total weight of fish caught by Emily -/
def total_fish_weight (trout_count catfish_count bluegill_count : ℕ)
                      (trout_weight catfish_weight bluegill_weight : ℝ) : ℝ :=
  (trout_count : ℝ) * trout_weight +
  (catfish_count : ℝ) * catfish_weight +
  (bluegill_count : ℝ) * bluegill_weight

/-- Proves that Emily caught 25 pounds of fish -/
theorem emily_fish_weight :
  total_fish_weight 4 3 5 2 1.5 2.5 = 25 := by
  sorry

end emily_fish_weight_l1783_178302


namespace cosine_amplitude_l1783_178303

/-- Given a cosine function y = a cos(bx + c) + d where a, b, c, d are positive constants,
    if the graph oscillates between 5 and 1, then a = 2. -/
theorem cosine_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_osc : ∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) :
  a = 2 := by
  sorry

end cosine_amplitude_l1783_178303


namespace exists_unique_affine_transformation_basis_exists_unique_affine_transformation_triangle_exists_unique_affine_transformation_parallelogram_l1783_178357

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define points and vectors
variable (O O' : V) (e1 e2 e1' e2' : V)
variable (A B C A1 B1 C1 : V)

-- Define affine transformation
def AffineTransformation (f : V → V) :=
  ∃ (T : V →L[ℝ] V) (b : V), ∀ x, f x = T x + b

-- Statement for part (a)
theorem exists_unique_affine_transformation_basis :
  ∃! f : V → V, AffineTransformation f ∧
  f O = O' ∧ f (O + e1) = O' + e1' ∧ f (O + e2) = O' + e2' :=
sorry

-- Statement for part (b)
theorem exists_unique_affine_transformation_triangle :
  ∃! f : V → V, AffineTransformation f ∧
  f A = A1 ∧ f B = B1 ∧ f C = C1 :=
sorry

-- Define parallelogram
def IsParallelogram (P Q R S : V) :=
  P - Q = S - R ∧ P - S = Q - R

-- Statement for part (c)
theorem exists_unique_affine_transformation_parallelogram
  (P Q R S P' Q' R' S' : V)
  (h1 : IsParallelogram P Q R S)
  (h2 : IsParallelogram P' Q' R' S') :
  ∃! f : V → V, AffineTransformation f ∧
  f P = P' ∧ f Q = Q' ∧ f R = R' ∧ f S = S' :=
sorry

end exists_unique_affine_transformation_basis_exists_unique_affine_transformation_triangle_exists_unique_affine_transformation_parallelogram_l1783_178357


namespace geometric_sequence_problem_l1783_178381

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a where a₄ = 7 and a₆ = 21, prove that a₈ = 63. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_a4 : a 4 = 7) 
  (h_a6 : a 6 = 21) : 
  a 8 = 63 := by
  sorry

end geometric_sequence_problem_l1783_178381


namespace division_problem_l1783_178339

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 190 →
  divisor = 21 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 9 := by
sorry

end division_problem_l1783_178339


namespace sqrt_expression_l1783_178342

theorem sqrt_expression : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := by
  sorry

end sqrt_expression_l1783_178342


namespace parabola_point_coordinates_l1783_178392

theorem parabola_point_coordinates (x y : ℝ) :
  y^2 = 4*x →                             -- P is on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 100 →                 -- Distance from P to focus (1, 0) is 10
  (x = 9 ∧ (y = 6 ∨ y = -6)) :=           -- Coordinates of P are (9, ±6)
by
  sorry

end parabola_point_coordinates_l1783_178392


namespace inequality_proof_l1783_178388

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x + y ≤ (y^2 / x) + (x^2 / y) := by
  sorry

end inequality_proof_l1783_178388


namespace quadratic_root_ratio_l1783_178384

theorem quadratic_root_ratio (c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (4 * x₁^2 - 5 * x₁ + c = 0) ∧ 
    (4 * x₂^2 - 5 * x₂ + c = 0) ∧ 
    (x₁ / x₂ = -3/4)) →
  c = -75 := by
sorry

end quadratic_root_ratio_l1783_178384


namespace maria_has_four_l1783_178305

/-- Represents a player in the card game -/
inductive Player : Type
  | Maria
  | Josh
  | Laura
  | Neil
  | Eva

/-- The score of each player -/
def score (p : Player) : ℕ :=
  match p with
  | Player.Maria => 13
  | Player.Josh => 15
  | Player.Laura => 9
  | Player.Neil => 18
  | Player.Eva => 19

/-- The set of all possible cards -/
def cards : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 12}

/-- Predicate to check if a pair of cards is valid for a player -/
def validCardPair (p : Player) (c1 c2 : ℕ) : Prop :=
  c1 ∈ cards ∧ c2 ∈ cards ∧ c1 + c2 = score p ∧ c1 ≠ c2

/-- Theorem stating that Maria must have received card number 4 -/
theorem maria_has_four :
  ∃ (c : ℕ), c ∈ cards ∧ c ≠ 4 ∧ validCardPair Player.Maria 4 c ∧
  (∀ (p : Player), p ≠ Player.Maria → ¬∃ (c1 c2 : ℕ), (c1 = 4 ∨ c2 = 4) ∧ validCardPair p c1 c2) :=
sorry

end maria_has_four_l1783_178305


namespace ellen_calorie_instruction_l1783_178361

/-- The total number of calories Ellen was instructed to eat in a day -/
def total_calories : ℕ := 2200

/-- The number of calories Ellen ate for breakfast -/
def breakfast_calories : ℕ := 353

/-- The number of calories Ellen had for lunch -/
def lunch_calories : ℕ := 885

/-- The number of calories Ellen had for afternoon snack -/
def snack_calories : ℕ := 130

/-- The number of calories Ellen has left for dinner -/
def dinner_calories : ℕ := 832

/-- Theorem stating that the total calories Ellen was instructed to eat
    is equal to the sum of all meals and snacks -/
theorem ellen_calorie_instruction :
  total_calories = breakfast_calories + lunch_calories + snack_calories + dinner_calories :=
by sorry

end ellen_calorie_instruction_l1783_178361


namespace inequality_upper_bound_upper_bound_tight_l1783_178332

theorem inequality_upper_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) ≤ 2 + Real.sqrt 5 := by
  sorry

theorem upper_bound_tight : 
  ∀ ε > 0, ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ 
  (2 + Real.sqrt 5) - (Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1)) < ε := by
  sorry

end inequality_upper_bound_upper_bound_tight_l1783_178332


namespace divide_decimals_l1783_178325

theorem divide_decimals : (0.08 : ℚ) / (0.002 : ℚ) = 40 := by
  sorry

end divide_decimals_l1783_178325


namespace rectangles_not_necessarily_similar_l1783_178347

-- Define a rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  length_positive : length > 0
  width_positive : width > 0

-- Define similarity for rectangles
def are_similar (r1 r2 : Rectangle) : Prop :=
  r1.length / r1.width = r2.length / r2.width

-- Theorem stating that rectangles are not necessarily similar
theorem rectangles_not_necessarily_similar :
  ∃ (r1 r2 : Rectangle), ¬(are_similar r1 r2) :=
sorry

end rectangles_not_necessarily_similar_l1783_178347


namespace same_solution_implies_a_equals_four_l1783_178324

theorem same_solution_implies_a_equals_four (a : ℝ) : 
  (∃ x : ℝ, 2 * x + 1 = 3 ∧ 2 - (a - x) / 3 = 1) → a = 4 := by
sorry

end same_solution_implies_a_equals_four_l1783_178324


namespace all_propositions_false_l1783_178380

-- Define the type for lines in space
def Line : Type := ℝ → ℝ → ℝ → Prop

-- Define the relations between lines
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry
def intersect (l1 l2 : Line) : Prop := sorry
def coplanar (l1 l2 : Line) : Prop := sorry

-- Define the propositions
def proposition1 (a b c : Line) : Prop :=
  (perpendicular a b ∧ perpendicular b c) → parallel a c

def proposition2 (a b c : Line) : Prop :=
  (skew a b ∧ skew b c) → skew a c

def proposition3 (a b c : Line) : Prop :=
  (intersect a b ∧ intersect b c) → intersect a c

def proposition4 (a b c : Line) : Prop :=
  (coplanar a b ∧ coplanar b c) → coplanar a c

-- Theorem stating that all propositions are false
theorem all_propositions_false (a b c : Line) :
  ¬ proposition1 a b c ∧
  ¬ proposition2 a b c ∧
  ¬ proposition3 a b c ∧
  ¬ proposition4 a b c :=
sorry

end all_propositions_false_l1783_178380


namespace line_through_point_with_equal_intercepts_l1783_178334

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def passesThroughPoint (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∃ (l : Line), passesThroughPoint l ⟨-3, -2⟩ ∧ hasEqualIntercepts l ∧
  ((l.a = 2 ∧ l.b = -3 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = 5)) := by
  sorry

end line_through_point_with_equal_intercepts_l1783_178334


namespace unique_integer_solution_l1783_178367

theorem unique_integer_solution : ∃! (x : ℕ+), (4 * x.val)^2 - 2 * x.val = 8066 := by
  sorry

end unique_integer_solution_l1783_178367


namespace infinite_product_value_l1783_178358

/-- The nth term of the sequence in the infinite product -/
def a (n : ℕ) : ℝ := (2^n)^(1 / 3^n)

/-- The sum of the exponents in the infinite product -/
noncomputable def S : ℝ := ∑' n, n / 3^n

/-- The infinite product -/
noncomputable def infiniteProduct : ℝ := 2^S

theorem infinite_product_value :
  infiniteProduct = 2^(3/4) := by sorry

end infinite_product_value_l1783_178358


namespace alice_needs_1615_stamps_l1783_178326

def stamps_problem (alice ernie peggy danny bert : ℕ) : Prop :=
  alice = 65 ∧
  danny = alice + 5 ∧
  peggy = 2 * danny ∧
  ernie = 3 * peggy ∧
  bert = 4 * ernie

theorem alice_needs_1615_stamps 
  (alice ernie peggy danny bert : ℕ) 
  (h : stamps_problem alice ernie peggy danny bert) : 
  bert - alice = 1615 := by
sorry

end alice_needs_1615_stamps_l1783_178326


namespace f_monotonicity_and_zeros_l1783_178370

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - x - a

theorem f_monotonicity_and_zeros (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ∨
  (∃ z : ℝ, ∀ x y : ℝ, x < y → x < z → f a x > f a y) ∧
    (∀ x y : ℝ, x < y → z < x → f a x < f a y) ∧
  (∃! x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ ≠ x₂) ↔ a ∈ Set.Ioo 0 1 ∪ Set.Ioi 1 :=
by sorry

end f_monotonicity_and_zeros_l1783_178370


namespace nth_monomial_formula_l1783_178319

/-- A sequence of monomials is defined as follows:
    For n = 1: 2x
    For n = 2: -4x^2
    For n = 3: 6x^3
    For n = 4: -8x^4
    For n = 5: 10x^5
    ...
    This function represents the coefficient of the nth monomial in the sequence. -/
def monomial_coefficient (n : ℕ) : ℤ :=
  (-1)^(n+1) * (2*n)

/-- This function represents the exponent of x in the nth monomial of the sequence. -/
def monomial_exponent (n : ℕ) : ℕ := n

/-- This theorem states that the nth monomial in the sequence
    can be expressed as (-1)^(n+1) * 2n * x^n for any positive integer n. -/
theorem nth_monomial_formula (n : ℕ) (h : n > 0) :
  monomial_coefficient n = (-1)^(n+1) * (2*n) ∧ monomial_exponent n = n :=
sorry

end nth_monomial_formula_l1783_178319


namespace units_digit_of_n_l1783_178363

/-- Given two natural numbers a and b, returns true if a has units digit 9 -/
def hasUnitsDigit9 (a : ℕ) : Prop :=
  a % 10 = 9

/-- Given a natural number n, returns its units digit -/
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_n (a b n : ℕ) 
  (h1 : a * b = 34^8) 
  (h2 : hasUnitsDigit9 a) 
  (h3 : n = b) : 
  unitsDigit n = 4 := by
sorry

end units_digit_of_n_l1783_178363


namespace shortest_distance_is_four_l1783_178372

/-- Represents the distances between three points A, B, and C. -/
structure TriangleDistances where
  ab : ℝ  -- Distance between A and B
  bc : ℝ  -- Distance between B and C
  ac : ℝ  -- Distance between A and C

/-- Given conditions for the problem -/
def problem_conditions (d : TriangleDistances) : Prop :=
  d.ab + d.bc = 10 ∧
  d.bc + d.ac = 13 ∧
  d.ac + d.ab = 11

/-- The theorem to be proved -/
theorem shortest_distance_is_four (d : TriangleDistances) 
  (h : problem_conditions d) : 
  min d.ab (min d.bc d.ac) = 4 := by
  sorry


end shortest_distance_is_four_l1783_178372


namespace ab_gt_1_sufficient_not_necessary_for_a_plus_b_gt_2_l1783_178360

theorem ab_gt_1_sufficient_not_necessary_for_a_plus_b_gt_2 :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y > 1 → x + y > 2) ∧
  (∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ c + d > 2 ∧ c * d ≤ 1) :=
by sorry

end ab_gt_1_sufficient_not_necessary_for_a_plus_b_gt_2_l1783_178360


namespace isabel_homework_problem_l1783_178333

theorem isabel_homework_problem (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) :
  math_pages = 2 →
  reading_pages = 4 →
  problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end isabel_homework_problem_l1783_178333


namespace stratified_sampling_group_c_l1783_178391

/-- Represents the number of cities selected from a group in a stratified sampling. -/
def citiesSelected (totalCities : ℕ) (sampleSize : ℕ) (groupSize : ℕ) : ℕ :=
  (sampleSize * groupSize) / totalCities

/-- Proves that in a stratified sampling of 6 cities from 24 total cities, 
    where 8 cities belong to group C, the number of cities selected from group C is 2. -/
theorem stratified_sampling_group_c : 
  citiesSelected 24 6 8 = 2 := by
  sorry

end stratified_sampling_group_c_l1783_178391


namespace sams_correct_percentage_l1783_178376

theorem sams_correct_percentage (y : ℕ) : 
  let total_problems := 8 * y
  let missed_problems := 3 * y
  let correct_problems := total_problems - missed_problems
  (correct_problems : ℚ) / (total_problems : ℚ) * 100 = 62.5 := by
  sorry

end sams_correct_percentage_l1783_178376


namespace program_arrangements_l1783_178304

/-- The number of solo segments in the program -/
def num_solo_segments : ℕ := 5

/-- The number of chorus segments in the program -/
def num_chorus_segments : ℕ := 3

/-- The number of spaces available for chorus segments after arranging solo segments -/
def num_spaces_for_chorus : ℕ := num_solo_segments + 1 - 1 -- +1 for spaces between solos, -1 for not placing first

/-- The number of different programs that can be arranged -/
def num_programs : ℕ := (Nat.factorial num_solo_segments) * (num_spaces_for_chorus.choose num_chorus_segments)

theorem program_arrangements :
  num_programs = 7200 :=
sorry

end program_arrangements_l1783_178304


namespace edge_tangent_sphere_radius_l1783_178369

/-- Represents a regular tetrahedron with associated spheres -/
structure RegularTetrahedron where
  r : ℝ  -- radius of inscribed sphere
  R : ℝ  -- radius of circumscribed sphere
  ρ : ℝ  -- radius of edge-tangent sphere
  h_r_pos : 0 < r
  h_R_pos : 0 < R
  h_ρ_pos : 0 < ρ
  h_R_r : R = 3 * r

/-- 
The radius of the sphere tangent to all edges of a regular tetrahedron 
is the geometric mean of the radii of its inscribed and circumscribed spheres 
-/
theorem edge_tangent_sphere_radius (t : RegularTetrahedron) : t.ρ^2 = t.R * t.r := by
  sorry

end edge_tangent_sphere_radius_l1783_178369


namespace triangle_area_l1783_178306

-- Define the linear functions
def f (x : ℝ) : ℝ := x - 4
def g (x : ℝ) : ℝ := -x - 4

-- Define the triangle
def Triangle := {(x, y) : ℝ × ℝ | (y = f x ∨ y = g x) ∧ y ≥ 0}

-- Theorem statement
theorem triangle_area : MeasureTheory.volume Triangle = 8 := by
  sorry

end triangle_area_l1783_178306


namespace inequality_preservation_l1783_178348

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end inequality_preservation_l1783_178348


namespace ultratown_block_perimeter_difference_l1783_178311

/-- Represents a rectangular city block with surrounding streets -/
structure CityBlock where
  length : ℝ
  width : ℝ
  street_width : ℝ

/-- Calculates the difference between outer and inner perimeters of a city block -/
def perimeter_difference (block : CityBlock) : ℝ :=
  2 * ((block.length + 2 * block.street_width) + (block.width + 2 * block.street_width)) -
  2 * (block.length + block.width)

/-- Theorem: The difference between outer and inner perimeters of the specified block is 200 feet -/
theorem ultratown_block_perimeter_difference :
  let block : CityBlock := {
    length := 500,
    width := 300,
    street_width := 25
  }
  perimeter_difference block = 200 := by
  sorry

end ultratown_block_perimeter_difference_l1783_178311


namespace fifteenth_term_of_sequence_l1783_178386

def arithmeticSequence (a₁ a₂ a₃ : ℕ) : ℕ → ℕ :=
  fun n => a₁ + (n - 1) * (a₂ - a₁)

theorem fifteenth_term_of_sequence (h : arithmeticSequence 3 14 25 3 = 25) :
  arithmeticSequence 3 14 25 15 = 157 := by
  sorry

end fifteenth_term_of_sequence_l1783_178386


namespace y_bound_l1783_178395

theorem y_bound (x y : ℝ) (hx : x = 7) (heq : (x - 2*y)^y = 0.001) : 
  0 < y ∧ y < 7/2 := by
  sorry

end y_bound_l1783_178395


namespace triangle_special_condition_l1783_178328

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and area S, if 4√3S = (a+b)² - c², then the measure of angle C is π/3 -/
theorem triangle_special_condition (a b c S : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (area_eq : 4 * Real.sqrt 3 * S = (a + b)^2 - c^2) :
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧ 
    A + B + C = π ∧
    S = 1/2 * a * b * Real.sin C ∧
    c^2 = a^2 + b^2 - 2*a*b*Real.cos C ∧
    C = π/3 :=
by sorry

end triangle_special_condition_l1783_178328


namespace odd_number_induction_l1783_178398

theorem odd_number_induction (P : ℕ → Prop) 
  (base : P 1) 
  (step : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) : 
  ∀ n : ℕ, n % 2 = 1 → P n := by
  sorry

end odd_number_induction_l1783_178398


namespace matthew_cooks_30_hotdogs_l1783_178331

/-- The number of hotdogs Matthew needs to cook for his family dinner -/
def total_hotdogs : ℝ :=
  let ella_emma := 2.5 * 2
  let luke := 2 * ella_emma
  let michael := 7
  let hunter := 1.5 * ella_emma
  let zoe := 0.5
  ella_emma + luke + michael + hunter + zoe

/-- Theorem stating that Matthew needs to cook 30 hotdogs -/
theorem matthew_cooks_30_hotdogs : total_hotdogs = 30 := by
  sorry

end matthew_cooks_30_hotdogs_l1783_178331


namespace sheridan_cats_l1783_178338

/-- The number of cats Mrs. Sheridan has after giving some away -/
def remaining_cats (initial : ℝ) (given_away : ℝ) : ℝ :=
  initial - given_away

/-- Proof that Mrs. Sheridan has 3.0 cats after giving away 14.0 cats -/
theorem sheridan_cats : remaining_cats 17.0 14.0 = 3.0 := by
  sorry

end sheridan_cats_l1783_178338


namespace intersection_of_M_and_N_l1783_178379

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = p.1^2}
def N : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {(1, 1), (-1, 1)} := by
sorry

end intersection_of_M_and_N_l1783_178379


namespace car_average_speed_l1783_178371

/-- Calculates the average speed of a car given specific conditions during a 4-hour trip -/
theorem car_average_speed : 
  let first_hour_speed : ℝ := 145
  let second_hour_speed : ℝ := 60
  let stop_duration : ℝ := 1/3
  let fourth_hour_min_speed : ℝ := 45
  let fourth_hour_max_speed : ℝ := 100
  let total_time : ℝ := 4 + stop_duration
  let fourth_hour_avg_speed : ℝ := (fourth_hour_min_speed + fourth_hour_max_speed) / 2
  let total_distance : ℝ := first_hour_speed + second_hour_speed + fourth_hour_avg_speed
  let average_speed : ℝ := total_distance / total_time
  ∃ ε > 0, |average_speed - 64.06| < ε :=
by
  sorry


end car_average_speed_l1783_178371


namespace vector_scalar_properties_l1783_178312

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_scalar_properties :
  (∀ (m : ℝ) (a b : V), m • (a - b) = m • a - m • b) ∧
  (∀ (m n : ℝ) (a : V), (m - n) • a = m • a - n • a) ∧
  (∃ (m : ℝ) (a b : V), m • a = m • b ∧ a ≠ b) ∧
  (∀ (m n : ℝ) (a : V), a ≠ 0 → m • a = n • a → m = n) :=
by sorry

end vector_scalar_properties_l1783_178312


namespace range_of_a_for_always_negative_l1783_178394

/-- The quadratic function f(x) = ax^2 + ax - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 4

/-- The predicate that f(x) < 0 for all real x -/
def always_negative (a : ℝ) : Prop := ∀ x, f a x < 0

/-- The theorem stating the range of a for which f(x) < 0 holds for all real x -/
theorem range_of_a_for_always_negative :
  ∀ a, always_negative a ↔ -16 < a ∧ a ≤ 0 :=
sorry

end range_of_a_for_always_negative_l1783_178394


namespace repeating_decimal_equals_fraction_l1783_178383

/-- The repeating decimal 0.3535... expressed as a real number -/
def repeating_decimal : ℚ := 35 / 99

theorem repeating_decimal_equals_fraction : repeating_decimal = 35 / 99 := by
  sorry

end repeating_decimal_equals_fraction_l1783_178383


namespace fraction_chain_l1783_178366

theorem fraction_chain (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 := by
sorry

end fraction_chain_l1783_178366


namespace vector_properties_and_projection_l1783_178313

/-- Given vectors in ℝ², prove properties about their relationships and projections -/
theorem vector_properties_and_projection :
  let a : ℝ × ℝ := (-1, 1)
  let b : ℝ × ℝ := (x, 3)
  let c : ℝ × ℝ := (5, y)
  let d : ℝ × ℝ := (8, 6)

  -- b is parallel to d
  (b.2 / b.1 = d.2 / d.1) →
  -- 4a + d is perpendicular to c
  ((4 * a.1 + d.1) * c.1 + (4 * a.2 + d.2) * c.2 = 0) →

  -- Prove that b and c have specific values
  (b = (4, 3) ∧ c = (5, -2)) ∧
  -- Prove that the projection of c onto a is -7√2/2
  (let proj := (a.1 * c.1 + a.2 * c.2) / Real.sqrt (a.1^2 + a.2^2)
   proj = -7 * Real.sqrt 2 / 2) :=
by sorry

end vector_properties_and_projection_l1783_178313


namespace cubic_roots_sum_l1783_178307

theorem cubic_roots_sum (a b c : ℝ) : 
  (3 * a^3 - 9 * a^2 + 54 * a - 12 = 0) →
  (3 * b^3 - 9 * b^2 + 54 * b - 12 = 0) →
  (3 * c^3 - 9 * c^2 + 54 * c - 12 = 0) →
  (a + 2*b - 2)^3 + (b + 2*c - 2)^3 + (c + 2*a - 2)^3 = 162 := by
sorry

end cubic_roots_sum_l1783_178307


namespace complex_equation_solution_l1783_178396

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h1 : i * i = -1) (h2 : z * (1 + i) = Complex.abs (2 * i)) : z = 1 - i := by
  sorry

end complex_equation_solution_l1783_178396


namespace correct_number_of_officers_l1783_178316

/-- Represents the number of officers in an office with given salary conditions. -/
def number_of_officers : ℕ :=
  let avg_salary_all : ℚ := 120
  let avg_salary_officers : ℚ := 420
  let avg_salary_non_officers : ℚ := 110
  let num_non_officers : ℕ := 450
  15

/-- Theorem stating that the number of officers is correct given the salary conditions. -/
theorem correct_number_of_officers :
  let avg_salary_all : ℚ := 120
  let avg_salary_officers : ℚ := 420
  let avg_salary_non_officers : ℚ := 110
  let num_non_officers : ℕ := 450
  let num_officers := number_of_officers
  (avg_salary_all * (num_officers + num_non_officers : ℚ) =
   avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers) :=
by sorry

end correct_number_of_officers_l1783_178316


namespace g_inverse_exists_g_inverse_composition_g_inverse_triple_composition_l1783_178330

def g : Fin 5 → Fin 5
  | 1 => 4
  | 2 => 3
  | 3 => 1
  | 4 => 5
  | 5 => 2

theorem g_inverse_exists : Function.Bijective g := sorry

theorem g_inverse_composition (x : Fin 5) : Function.invFun g (g x) = x := sorry

theorem g_inverse_triple_composition :
  Function.invFun g (Function.invFun g (Function.invFun g 3)) = 4 := by
  sorry

end g_inverse_exists_g_inverse_composition_g_inverse_triple_composition_l1783_178330


namespace point_order_l1783_178351

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the line y = -7x + 14 -/
def lies_on_line (p : Point) : Prop :=
  p.y = -7 * p.x + 14

theorem point_order (A B C : Point) 
  (hA : lies_on_line A) 
  (hB : lies_on_line B) 
  (hC : lies_on_line C) 
  (hx : A.x > C.x ∧ C.x > B.x) : 
  A.y < C.y ∧ C.y < B.y := by
  sorry

end point_order_l1783_178351


namespace share_division_l1783_178375

theorem share_division (total : ℝ) (a b c : ℝ) 
  (h_total : total = 400)
  (h_sum : a + b + c = total)
  (h_a : a = (2/3) * (b + c))
  (h_b : b = (6/9) * (a + c)) :
  a = 160 := by sorry

end share_division_l1783_178375


namespace power_of_8_mod_100_l1783_178349

theorem power_of_8_mod_100 : 8^2023 % 100 = 12 := by
  sorry

end power_of_8_mod_100_l1783_178349


namespace count_positive_numbers_l1783_178390

theorem count_positive_numbers : 
  let numbers : List ℚ := [-3, -1, 1/3, 0, -3/7, 2017]
  (numbers.filter (λ x => x > 0)).length = 2 := by
  sorry

end count_positive_numbers_l1783_178390


namespace binomial_sum_equality_l1783_178359

theorem binomial_sum_equality (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) : 
  (∑' d, Nat.choose (n - r + 1) d * Nat.choose (r - 1) (d - 1)) = Nat.choose n r :=
by sorry

end binomial_sum_equality_l1783_178359
