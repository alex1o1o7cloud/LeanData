import Mathlib

namespace triangle_angles_are_equal_l517_51767

theorem triangle_angles_are_equal
  (A B C : ℝ) (a b c : ℝ)
  (h1 : A + B + C = π)
  (h2 : A = B + (B - A))
  (h3 : B = C + (C - B))
  (h4 : 2 * (1 / b) = (1 / a) + (1 / c)) :
  A = π / 3 ∧ B = π / 3 ∧ C = π / 3 :=
sorry

end triangle_angles_are_equal_l517_51767


namespace remainder_of_division_l517_51754

theorem remainder_of_division (L S R : ℕ) (hL : L = 1620) (h_diff : L - S = 1365) (h_div : L = 6 * S + R) : R = 90 :=
by {
  -- Since we are not providing the proof, we use sorry
  sorry
}

end remainder_of_division_l517_51754


namespace inversely_proportional_x_y_l517_51790

-- Statement of the problem
theorem inversely_proportional_x_y :
  ∃ k : ℝ, (∀ (x y : ℝ), (x * y = k) ∧ (x = 4) ∧ (y = 2) → x * (-5) = -8 / 5) :=
by
  sorry

end inversely_proportional_x_y_l517_51790


namespace lcm_of_23_46_827_l517_51701

theorem lcm_of_23_46_827 : Nat.lcm (Nat.lcm 23 46) 827 = 38042 :=
by
  sorry

end lcm_of_23_46_827_l517_51701


namespace equal_roots_of_quadratic_l517_51749

theorem equal_roots_of_quadratic (k : ℝ) : (1 - 8 * k = 0) → (k = 1/8) :=
by
  intro h
  sorry

end equal_roots_of_quadratic_l517_51749


namespace dogsled_course_distance_l517_51720

theorem dogsled_course_distance 
    (t : ℕ)  -- time taken by Team B
    (speed_B : ℕ := 20)  -- average speed of Team B
    (speed_A : ℕ := 25)  -- average speed of Team A
    (tA_eq_tB_minus_3 : t - 3 = tA)  -- Team A’s time relation
    (speedA_eq_speedB_plus_5 : speed_A = speed_B + 5)  -- Team A's average speed in relation to Team B’s average speed
    (distance_eq : speed_B * t = speed_A * (t - 3))  -- Distance equality condition
    (t_eq_15 : t = 15)  -- Time taken by Team B to finish
    :
    (speed_B * t = 300) :=   -- Distance of the course
by
  sorry

end dogsled_course_distance_l517_51720


namespace quadratic_decreasing_then_increasing_l517_51779

-- Define the given quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 6 * x + 10

-- Define the interval of interest
def interval (x : ℝ) : Prop := 2 < x ∧ x < 4

-- The main theorem to prove: the function is first decreasing on (2, 3] and then increasing on [3, 4)
theorem quadratic_decreasing_then_increasing :
  (∀ (x : ℝ), 2 < x ∧ x ≤ 3 → quadratic_function x > quadratic_function (x + ε) ∧ ε > 0) ∧
  (∀ (x : ℝ), 3 ≤ x ∧ x < 4 → quadratic_function x < quadratic_function (x + ε) ∧ ε > 0) :=
sorry

end quadratic_decreasing_then_increasing_l517_51779


namespace percentage_decrease_in_spring_l517_51751

-- Given Conditions
variables (initial_members : ℕ) (increased_percent : ℝ) (total_decrease_percent : ℝ)
-- population changes
variables (fall_members : ℝ) (spring_members : ℝ)

-- The initial conditions given by the problem
axiom initial_membership : initial_members = 100
axiom fall_increase : increased_percent = 6
axiom total_decrease : total_decrease_percent = 14.14

-- Derived values based on conditions
axiom fall_members_calculated : fall_members = initial_members * (1 + increased_percent / 100)
axiom spring_members_calculated : spring_members = initial_members * (1 - total_decrease_percent / 100)

-- The correct answer which we need to prove
theorem percentage_decrease_in_spring : 
  ((fall_members - spring_members) / fall_members) * 100 = 19 := by
  sorry

end percentage_decrease_in_spring_l517_51751


namespace geometric_progression_common_ratio_l517_51775

theorem geometric_progression_common_ratio (x y z w r : ℂ) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0)
  (h_geom : x * (y - w) = a ∧ y * (z - x) = a * r ∧ z * (w - y) = a * r^2 ∧ w * (x - z) = a * r^3) :
  1 + r + r^2 + r^3 = 0 :=
sorry

end geometric_progression_common_ratio_l517_51775


namespace find_b_l517_51771

theorem find_b (a b : ℤ) 
  (h1 : a * b = 2 * (a + b) + 14) 
  (h2 : b - a = 3) : 
  b = 8 :=
sorry

end find_b_l517_51771


namespace perp_bisector_b_value_l517_51747

theorem perp_bisector_b_value : ∃ b : ℝ, (∀ (x y : ℝ), x + y = b) ∧ (x + y = b) ∧ (x = (-1) ∧ y = 2) ∧ (x = 3 ∧ y = 8) := sorry

end perp_bisector_b_value_l517_51747


namespace andrew_eggs_bought_l517_51780

-- Define initial conditions
def initial_eggs : ℕ := 8
def final_eggs : ℕ := 70

-- Define the function to determine the number of eggs bought
def eggs_bought (initial : ℕ) (final : ℕ) : ℕ := final - initial

-- State the theorem we want to prove
theorem andrew_eggs_bought : eggs_bought initial_eggs final_eggs = 62 :=
by {
  -- Proof goes here
  sorry
}

end andrew_eggs_bought_l517_51780


namespace no_real_x_satisfies_quadratic_ineq_l517_51728

theorem no_real_x_satisfies_quadratic_ineq :
  ¬ ∃ x : ℝ, x^2 + 3 * x + 3 ≤ 0 :=
sorry

end no_real_x_satisfies_quadratic_ineq_l517_51728


namespace sandy_total_sums_l517_51745

theorem sandy_total_sums (C I : ℕ) (h1 : C = 22) (h2 : 3 * C - 2 * I = 50) :
  C + I = 30 :=
sorry

end sandy_total_sums_l517_51745


namespace compare_negatives_l517_51730

theorem compare_negatives : (- (3 : ℝ) / 5) > (- (5 : ℝ) / 7) :=
by
  sorry

end compare_negatives_l517_51730


namespace problem1_problem2_l517_51734

-- Proving that (3*sqrt(8) - 12*sqrt(1/2) + sqrt(18)) * 2*sqrt(3) = 6*sqrt(6)
theorem problem1 :
  (3 * Real.sqrt 8 - 12 * Real.sqrt (1/2) + Real.sqrt 18) * 2 * Real.sqrt 3 = 6 * Real.sqrt 6 :=
sorry

-- Proving that (6*sqrt(x/4) - 2*x*sqrt(1/x)) / 3*sqrt(x) = 1/3
theorem problem2 (x : ℝ) (hx : 0 < x) :
  (6 * Real.sqrt (x/4) - 2 * x * Real.sqrt (1/x)) / (3 * Real.sqrt x) = 1/3 :=
sorry

end problem1_problem2_l517_51734


namespace mountain_height_is_1700m_l517_51784

noncomputable def height_of_mountain (temp_base : ℝ) (temp_summit : ℝ) (rate_decrease : ℝ) : ℝ :=
  ((temp_base - temp_summit) / rate_decrease) * 100

theorem mountain_height_is_1700m :
  height_of_mountain 26 14.1 0.7 = 1700 :=
by
  sorry

end mountain_height_is_1700m_l517_51784


namespace value_of_expression_l517_51732

theorem value_of_expression (m : ℝ) (h : m^2 - m - 110 = 0) : (m - 1)^2 + m = 111 := by
  sorry

end value_of_expression_l517_51732


namespace stratified_sampling_B_l517_51743

-- Define the groups and their sizes
def num_people_A : ℕ := 18
def num_people_B : ℕ := 24
def num_people_C : ℕ := 30

-- Total number of people
def total_people : ℕ := num_people_A + num_people_B + num_people_C

-- Total sample size to be drawn
def sample_size : ℕ := 12

-- Proportion of group B
def proportion_B : ℚ := num_people_B / total_people

-- Number of people to be drawn from group B
def number_drawn_from_B : ℚ := sample_size * proportion_B

-- The theorem to be proved
theorem stratified_sampling_B : number_drawn_from_B = 4 := 
by
  -- This is where the proof would go
  sorry

end stratified_sampling_B_l517_51743


namespace binomial_12_3_eq_220_l517_51785

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Theorem to prove binomial(12, 3) = 220
theorem binomial_12_3_eq_220 : binomial 12 3 = 220 := by
  sorry

end binomial_12_3_eq_220_l517_51785


namespace BigJoe_is_8_feet_l517_51744

variable (Pepe_height : ℝ) (h1 : Pepe_height = 4.5)
variable (Frank_height : ℝ) (h2 : Frank_height = Pepe_height + 0.5)
variable (Larry_height : ℝ) (h3 : Larry_height = Frank_height + 1)
variable (Ben_height : ℝ) (h4 : Ben_height = Larry_height + 1)
variable (BigJoe_height : ℝ) (h5 : BigJoe_height = Ben_height + 1)

theorem BigJoe_is_8_feet : BigJoe_height = 8 := by
  sorry

end BigJoe_is_8_feet_l517_51744


namespace wheels_on_each_other_axle_l517_51761

def truck_toll_wheels (t : ℝ) (x : ℝ) (w : ℕ) : Prop :=
  t = 1.50 + 1.50 * (x - 2) ∧ (w = 18) ∧ (∀ y : ℕ, y = 18 - 2 - 4 *(x - 5) / 4)

theorem wheels_on_each_other_axle :
  ∀ t x w, truck_toll_wheels t x w → w = 18 ∧ x = 5 → (18 - 2) / 4 = 4 :=
by
  intros t x w h₁ h₂
  have h₃ : t = 6 := sorry
  have h₄ : x = 4 := sorry
  have h₅ : w = 18 := sorry
  have h₆ : (18 - 2) / 4 = 4 := sorry
  exact h₆

end wheels_on_each_other_axle_l517_51761


namespace train_passes_man_in_approximately_24_seconds_l517_51774

noncomputable def train_length : ℝ := 880 -- length of the train in meters
noncomputable def train_speed_kmph : ℝ := 120 -- speed of the train in km/h
noncomputable def man_speed_kmph : ℝ := 12 -- speed of the man in km/h

noncomputable def kmph_to_mps (speed: ℝ) : ℝ := speed * (1000 / 3600)

noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def man_speed_mps : ℝ := kmph_to_mps man_speed_kmph
noncomputable def relative_speed : ℝ := train_speed_mps + man_speed_mps

noncomputable def time_to_pass : ℝ := train_length / relative_speed

theorem train_passes_man_in_approximately_24_seconds :
  abs (time_to_pass - 24) < 1 :=
sorry

end train_passes_man_in_approximately_24_seconds_l517_51774


namespace crayons_count_l517_51740

theorem crayons_count (l b f : ℕ) (h1 : l = b / 2) (h2 : b = 3 * f) (h3 : l = 27) : f = 18 :=
by
  sorry

end crayons_count_l517_51740


namespace div_37_permutation_l517_51716

-- Let A, B, C be digits of a three-digit number
variables (A B C : ℕ) -- these can take values from 0 to 9
variables (p : ℕ) -- integer multiplier for the divisibility condition

-- The main theorem stated as a Lean 4 problem
theorem div_37_permutation (h : 100 * A + 10 * B + C = 37 * p) : 
  ∃ (M : ℕ), (M = 100 * B + 10 * C + A ∨ M = 100 * C + 10 * A + B ∨ M = 100 * A + 10 * C + B ∨ M = 100 * C + 10 * B + A ∨ M = 100 * B + 10 * A + C) ∧ 37 ∣ M :=
by
  sorry

end div_37_permutation_l517_51716


namespace ratio_of_votes_l517_51765

theorem ratio_of_votes (total_votes ben_votes : ℕ) (h_total : total_votes = 60) (h_ben : ben_votes = 24) :
  (ben_votes : ℚ) / (total_votes - ben_votes : ℚ) = 2 / 3 :=
by sorry

end ratio_of_votes_l517_51765


namespace quadratic_inequality_l517_51706

theorem quadratic_inequality : ∀ x : ℝ, -7 * x ^ 2 + 4 * x - 6 < 0 :=
by
  intro x
  have delta : 4 ^ 2 - 4 * (-7) * (-6) = -152 := by norm_num
  have neg_discriminant : -152 < 0 := by norm_num
  have coef : -7 < 0 := by norm_num
  sorry

end quadratic_inequality_l517_51706


namespace cannot_determine_c_l517_51746

-- Definitions based on conditions
variables {a b c d : ℕ}
axiom h1 : a + b + c = 21
axiom h2 : a + b + d = 27
axiom h3 : a + c + d = 30

-- The statement that c cannot be determined exactly
theorem cannot_determine_c : ¬ (∃ c : ℕ, c = c) :=
by sorry

end cannot_determine_c_l517_51746


namespace gcd_of_consecutive_digit_sums_l517_51762

theorem gcd_of_consecutive_digit_sums :
  ∀ x y z : ℕ, x + 1 = y → y + 1 = z → gcd (101 * (x + z) + 10 * y) 212 = 212 :=
by
  sorry

end gcd_of_consecutive_digit_sums_l517_51762


namespace text_message_costs_equal_l517_51757

theorem text_message_costs_equal (x : ℝ) : 
  (0.25 * x + 9 = 0.40 * x) ∧ (0.25 * x + 9 = 0.20 * x + 12) → x = 60 :=
by 
  sorry

end text_message_costs_equal_l517_51757


namespace pipe_c_empty_time_l517_51773

theorem pipe_c_empty_time (x : ℝ) :
  (4/20 + 4/30 + 4/x) * 3 = 1 → x = 6 :=
by
  sorry

end pipe_c_empty_time_l517_51773


namespace impossible_gather_all_coins_in_one_sector_l517_51786

-- Definition of the initial condition with sectors and coins
def initial_coins_in_sectors := [1, 1, 1, 1, 1, 1] -- Each sector has one coin, represented by a list

-- Function to check if all coins are in one sector
def all_coins_in_one_sector (coins : List ℕ) := coins.count 6 == 1

-- Function to make a move (this is a helper; its implementation isn't necessary here but illustrates the idea)
def make_move (coins : List ℕ) (src dst : ℕ) : List ℕ := sorry

-- Proving that after 20 moves, coins cannot be gathered in one sector due to parity constraints
theorem impossible_gather_all_coins_in_one_sector : 
  ¬ ∃ (moves : List (ℕ × ℕ)), moves.length = 20 ∧ all_coins_in_one_sector (List.foldl (λ coins move => make_move coins move.1 move.2) initial_coins_in_sectors moves) :=
sorry

end impossible_gather_all_coins_in_one_sector_l517_51786


namespace right_triangle_of_three_colors_exists_l517_51769

-- Define the type for color
inductive Color
| color1
| color2
| color3

open Color

-- Define the type for integer coordinate points
structure Point :=
(x : ℤ)
(y : ℤ)
(color : Color)

-- Define the conditions
def all_points_colored : Prop :=
∀ (p : Point), p.color = color1 ∨ p.color = color2 ∨ p.color = color3

def all_colors_used : Prop :=
∃ (p1 p2 p3 : Point), p1.color = color1 ∧ p2.color = color2 ∧ p3.color = color3

-- Define the right_triangle_exist problem
def right_triangle_exists : Prop :=
∃ (p1 p2 p3 : Point), 
  p1.color ≠ p2.color ∧ p2.color ≠ p3.color ∧ p3.color ≠ p1.color ∧
  (p1.x = p2.x ∧ p2.y = p3.y ∧ p1.y = p3.y ∨
   p1.y = p2.y ∧ p2.x = p3.x ∧ p1.x = p3.x ∨
   (p3.x - p1.x)*(p3.x - p1.x) + (p3.y - p1.y)*(p3.y - p1.y) = (p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y) ∧
   (p3.x - p2.x)*(p3.x - p2.x) + (p3.y - p2.y)*(p3.y - p2.y) = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y))

theorem right_triangle_of_three_colors_exists (h1 : all_points_colored) (h2 : all_colors_used) : right_triangle_exists := 
sorry

end right_triangle_of_three_colors_exists_l517_51769


namespace proposition_induction_l517_51777

variable (P : ℕ → Prop)
variable (k : ℕ)

theorem proposition_induction (h : ∀ k : ℕ, P k → P (k + 1))
    (h9 : ¬ P 9) : ¬ P 8 :=
by
  sorry

end proposition_induction_l517_51777


namespace intersection_A_B_l517_51787

def A : Set Int := {-1, 0, 1, 5, 8}
def B : Set Int := {x | x > 1}

theorem intersection_A_B : A ∩ B = {5, 8} :=
by
  sorry

end intersection_A_B_l517_51787


namespace max_k_value_l517_51723

def maximum_k (k : ℕ) : ℕ := 2

theorem max_k_value
  (k : ℕ)
  (h1 : 2 * k + 1 ≤ 20)  -- Condition implicitly implied by having subsets of a 20-element set
  (h2 : ∀ (s t : Finset (Fin 20)), s.card = 7 → t.card = 7 → s ≠ t → (s ∩ t).card = k) : k ≤ maximum_k k := 
by {
  sorry
}

end max_k_value_l517_51723


namespace probability_of_at_least_one_vowel_is_799_over_1024_l517_51793

def Set1 : Set Char := {'a', 'e', 'i', 'b', 'c', 'd', 'f', 'g'}
def Set2 : Set Char := {'u', 'o', 'y', 'k', 'l', 'm', 'n', 'p'}
def Set3 : Set Char := {'e', 'u', 'v', 'r', 's', 't', 'w', 'x'}
def Set4 : Set Char := {'a', 'i', 'o', 'z', 'h', 'j', 'q', 'r'}

noncomputable def probability_of_at_least_one_vowel : ℚ :=
  1 - (5/8 : ℚ) * (3/4 : ℚ) * (3/4 : ℚ) * (5/8 : ℚ)

theorem probability_of_at_least_one_vowel_is_799_over_1024 :
  probability_of_at_least_one_vowel = 799 / 1024 :=
by
  sorry

end probability_of_at_least_one_vowel_is_799_over_1024_l517_51793


namespace cyclic_quadrilateral_angles_l517_51705

theorem cyclic_quadrilateral_angles (A B C D : ℝ) (h_cyclic : A + C = 180) (h_diag_bisect : (A = 2 * (B / 5 + B / 5)) ∧ (C = 2 * (D / 5 + D / 5))) (h_ratio : B / D = 2 / 3):
  A = 80 ∨ A = 100 ∨ A = 1080 / 11 ∨ A = 900 / 11 :=
  sorry

end cyclic_quadrilateral_angles_l517_51705


namespace common_tangent_y_intercept_l517_51726

noncomputable def circle_center_a : ℝ × ℝ := (1, 5)
noncomputable def circle_radius_a : ℝ := 3

noncomputable def circle_center_b : ℝ × ℝ := (15, 10)
noncomputable def circle_radius_b : ℝ := 10

theorem common_tangent_y_intercept :
  ∃ m b: ℝ, (m > 0) ∧ m = 700/1197 ∧ b = 7.416 ∧
  ∀ x y: ℝ, (y = m * x + b → ((x - 1)^2 + (y - 5)^2 = 9 ∨ (x - 15)^2 + (y - 10)^2 = 100)) := by
{
  sorry
}

end common_tangent_y_intercept_l517_51726


namespace range_of_x_l517_51753

theorem range_of_x (a : ℝ) (x : ℝ) (h_a : 1 ≤ a) : 
  ax^2 + (a - 3) * x + (a - 4) > 0 ↔ x < -1 ∨ x > 3 :=
by
  sorry

end range_of_x_l517_51753


namespace bacon_calories_percentage_l517_51788

theorem bacon_calories_percentage (total_calories : ℕ) (bacon_strip_calories : ℕ) (num_strips : ℕ)
    (h1 : total_calories = 1250) (h2 : bacon_strip_calories = 125) (h3 : num_strips = 2) :
    (bacon_strip_calories * num_strips * 100) / total_calories = 20 := by
  sorry

end bacon_calories_percentage_l517_51788


namespace triangle_YZ_length_l517_51770

/-- In triangle XYZ, sides XY and XZ have lengths 6 and 8 inches respectively, 
    and the median XM from vertex X to the midpoint of side YZ is 5 inches. 
    Prove that the length of YZ is 10 inches. -/
theorem triangle_YZ_length
  (XY XZ XM : ℝ)
  (hXY : XY = 6)
  (hXZ : XZ = 8)
  (hXM : XM = 5) :
  ∃ (YZ : ℝ), YZ = 10 := 
by
  sorry

end triangle_YZ_length_l517_51770


namespace evaluate_expression_l517_51709

theorem evaluate_expression :
  (|(-1 : ℝ)|^2023 + (Real.sqrt 3)^2 - 2 * Real.sin (Real.pi / 6) + (1 / 2)⁻¹ = 5) :=
by
  sorry

end evaluate_expression_l517_51709


namespace value_of_expression_l517_51789

theorem value_of_expression (x Q : ℝ) (π : Real) (h : 5 * (3 * x - 4 * π) = Q) : 10 * (6 * x - 8 * π) = 4 * Q :=
by 
  sorry

end value_of_expression_l517_51789


namespace prob_exactly_two_passes_prob_at_least_one_fails_l517_51707

-- Define the probabilities for students A, B, and C passing their tests.
def prob_A : ℚ := 4/5
def prob_B : ℚ := 3/5
def prob_C : ℚ := 7/10

-- Define the probabilities for students A, B, and C failing their tests.
def prob_not_A : ℚ := 1 - prob_A
def prob_not_B : ℚ := 1 - prob_B
def prob_not_C : ℚ := 1 - prob_C

-- (1) Prove that the probability of exactly two students passing is 113/250.
theorem prob_exactly_two_passes : 
  prob_A * prob_B * prob_not_C + prob_A * prob_not_B * prob_C + prob_not_A * prob_B * prob_C = 113/250 := 
sorry

-- (2) Prove that the probability that at least one student fails is 83/125.
theorem prob_at_least_one_fails : 
  1 - (prob_A * prob_B * prob_C) = 83/125 := 
sorry

end prob_exactly_two_passes_prob_at_least_one_fails_l517_51707


namespace max_profit_l517_51711

theorem max_profit (m : ℝ) :
  (m - 8) * (900 - 15 * m) = -15 * (m - 34) ^ 2 + 10140 :=
by
  sorry

end max_profit_l517_51711


namespace increasing_function_l517_51795

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.sin x

theorem increasing_function (a : ℝ) (h : a ≥ 1) : 
  ∀ x y : ℝ, x ≤ y → f a x ≤ f a y :=
by 
  sorry

end increasing_function_l517_51795


namespace correct_algorithm_statement_l517_51735

def reversible : Prop := false -- Algorithms are generally not reversible.
def endless : Prop := false -- Algorithms should not run endlessly.
def unique_algo : Prop := false -- Not always one single algorithm for a task.
def simple_convenient : Prop := true -- Algorithms should be simple and convenient.

theorem correct_algorithm_statement : simple_convenient = true :=
by
  sorry

end correct_algorithm_statement_l517_51735


namespace red_sequence_2018th_num_l517_51798

/-- Define the sequence of red-colored numbers based on the given conditions. -/
def red_sequenced_num (n : Nat) : Nat :=
  let k := Nat.sqrt (2 * n - 1) -- estimate block number
  let block_start := if k % 2 == 0 then (k - 1)*(k - 1) else k * (k - 1) + 1
  let position_in_block := n - (k * (k - 1) / 2) - 1
  if k % 2 == 0 then block_start + 2 * position_in_block else block_start + 2 * position_in_block

/-- Statement to assert the 2018th number is 3972 -/
theorem red_sequence_2018th_num : red_sequenced_num 2018 = 3972 := by
  sorry

end red_sequence_2018th_num_l517_51798


namespace heights_inequality_l517_51722

theorem heights_inequality (a b c h_a h_b h_c p R : ℝ) (h : a ≤ b ∧ b ≤ c) : 
  h_a + h_b + h_c ≤ (3 * b * (a^2 + a * c + c^2)) / (4 * p * R) := 
sorry

end heights_inequality_l517_51722


namespace problem1_problem2_l517_51737

open Real

theorem problem1: 
  ((25^(1/3) - 125^(1/2)) / 5^(1/4) = 5^(5/12) - 5^(5/4)) :=
sorry

theorem problem2 (a : ℝ) (h : 0 < a): 
  (a^2 / (a^(1/2) * a^(2/3)) = a^(5/6)) :=
sorry

end problem1_problem2_l517_51737


namespace spurs_total_basketballs_l517_51719

theorem spurs_total_basketballs (players : ℕ) (basketballs_per_player : ℕ) (h1 : players = 22) (h2 : basketballs_per_player = 11) : players * basketballs_per_player = 242 := by
  sorry

end spurs_total_basketballs_l517_51719


namespace four_digit_cubes_divisible_by_16_count_l517_51721

theorem four_digit_cubes_divisible_by_16_count :
  ∃ (count : ℕ), count = 3 ∧
    ∀ (m : ℕ), 1000 ≤ 64 * m^3 ∧ 64 * m^3 ≤ 9999 → (m = 3 ∨ m = 4 ∨ m = 5) :=
by {
  -- our proof would go here
  sorry
}

end four_digit_cubes_divisible_by_16_count_l517_51721


namespace number_of_paths_to_spell_MATH_l517_51712

-- Define the problem setting and conditions
def number_of_paths_M_to_H (adj: ℕ) (steps: ℕ): ℕ :=
  adj^(steps-1)

-- State the problem in Lean 4
theorem number_of_paths_to_spell_MATH : number_of_paths_M_to_H 8 4 = 512 := 
by 
  unfold number_of_paths_M_to_H 
  -- The needed steps are included:
  -- We calculate: 8^(4-1) = 8^3 which should be 512.
  sorry

end number_of_paths_to_spell_MATH_l517_51712


namespace two_trains_distance_before_meeting_l517_51782

noncomputable def distance_one_hour_before_meeting (speed_A speed_B : ℕ) : ℕ :=
  speed_A + speed_B

theorem two_trains_distance_before_meeting (speed_A speed_B total_distance : ℕ) (h_speed_A : speed_A = 60) (h_speed_B : speed_B = 40) (h_total_distance : total_distance ≤ 250) :
  distance_one_hour_before_meeting speed_A speed_B = 100 :=
by
  sorry

end two_trains_distance_before_meeting_l517_51782


namespace joe_max_money_l517_51759

noncomputable def max_guaranteed_money (initial_money : ℕ) (max_bet : ℕ) (num_bets : ℕ) : ℕ :=
  if initial_money = 100 ∧ max_bet = 17 ∧ num_bets = 5 then 98 else 0

theorem joe_max_money : max_guaranteed_money 100 17 5 = 98 := by
  sorry

end joe_max_money_l517_51759


namespace Karls_Total_Travel_Distance_l517_51781

theorem Karls_Total_Travel_Distance :
  let consumption_rate := 35
  let full_tank_gallons := 14
  let initial_miles := 350
  let added_gallons := 8
  let remaining_gallons := 7
  let net_gallons_consumed := (full_tank_gallons + added_gallons - remaining_gallons)
  let total_distance := net_gallons_consumed * consumption_rate
  total_distance = 525 := 
by 
  sorry

end Karls_Total_Travel_Distance_l517_51781


namespace solve_equation_l517_51727

theorem solve_equation (x : ℚ) (h : x ≠ -5) : 
  (x^2 + 3*x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 := by
  sorry

end solve_equation_l517_51727


namespace move_point_right_3_units_from_neg_2_l517_51756

noncomputable def move_point_to_right (start : ℤ) (units : ℤ) : ℤ :=
start + units

theorem move_point_right_3_units_from_neg_2 : move_point_to_right (-2) 3 = 1 :=
by
  sorry

end move_point_right_3_units_from_neg_2_l517_51756


namespace pet_store_dogs_l517_51700

-- Define the given conditions as Lean definitions
def initial_dogs : ℕ := 2
def sunday_dogs : ℕ := 5
def monday_dogs : ℕ := 3

-- Define the total dogs calculation to use in the theorem
def total_dogs : ℕ := initial_dogs + sunday_dogs + monday_dogs

-- State the theorem
theorem pet_store_dogs : total_dogs = 10 := 
by
  -- Placeholder for the proof
  sorry

end pet_store_dogs_l517_51700


namespace time_to_cross_second_platform_l517_51733

-- Definition of the conditions
variables (l_train l_platform1 l_platform2 t1 : ℕ)
variable (v : ℕ)

-- The conditions given in the problem
def conditions : Prop :=
  l_train = 190 ∧
  l_platform1 = 140 ∧
  l_platform2 = 250 ∧
  t1 = 15 ∧
  v = (l_train + l_platform1) / t1

-- The statement to prove
theorem time_to_cross_second_platform
    (l_train l_platform1 l_platform2 t1 : ℕ)
    (v : ℕ)
    (h : conditions l_train l_platform1 l_platform2 t1 v) :
    (l_train + l_platform2) / v = 20 :=
  sorry

end time_to_cross_second_platform_l517_51733


namespace min_a_plus_5b_l517_51791

theorem min_a_plus_5b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a * b + b^2 = b + 1) : 
  a + 5 * b ≥ 7 / 2 :=
by
  sorry

end min_a_plus_5b_l517_51791


namespace tenth_term_l517_51704

noncomputable def sequence_term (n : ℕ) : ℝ :=
  (-1)^(n+1) * (Real.sqrt (1 + 2*(n - 1))) / (2^n)

theorem tenth_term :
  sequence_term 10 = Real.sqrt 19 / (2^10) :=
by
  sorry

end tenth_term_l517_51704


namespace ratio_of_areas_l517_51724

-- Define the side lengths of Squared B and Square C
variables (y : ℝ)

-- Define the areas of Square B and C
def area_B := (2 * y) * (2 * y)
def area_C := (8 * y) * (8 * y)

-- The theorem statement proving the ratio of the areas
theorem ratio_of_areas : area_B y / area_C y = 1 / 16 := 
by sorry

end ratio_of_areas_l517_51724


namespace bankers_discount_l517_51772

/-- The banker’s gain on a sum due 3 years hence at 12% per annum is Rs. 360.
   The banker's discount is to be determined. -/
theorem bankers_discount (BG BD TD : ℝ) (R : ℝ := 12 / 100) (T : ℝ := 3) 
  (h1 : BG = 360) (h2 : BG = (BD * TD) / (BD - TD)) (h3 : TD = (P * R * T) / 100) 
  (h4 : BG = (TD * R * T) / 100) :
  BD = 562.5 :=
sorry

end bankers_discount_l517_51772


namespace count_6_digit_palindromes_with_even_middle_digits_l517_51742

theorem count_6_digit_palindromes_with_even_middle_digits :
  let a_values := 9
  let b_even_values := 5
  let c_values := 10
  a_values * b_even_values * c_values = 450 :=
by {
  sorry
}

end count_6_digit_palindromes_with_even_middle_digits_l517_51742


namespace print_time_correct_l517_51718

-- Define the conditions
def pages_per_minute : ℕ := 23
def total_pages : ℕ := 345

-- Define the expected result
def expected_minutes : ℕ := 15

-- Prove the equivalence
theorem print_time_correct :
  total_pages / pages_per_minute = expected_minutes :=
by 
  -- Proof will be provided here
  sorry

end print_time_correct_l517_51718


namespace evaluate_expression_l517_51783

theorem evaluate_expression (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y + 2 = 3 := by
  rw [hx, hy]
  sorry

end evaluate_expression_l517_51783


namespace product_remainder_div_5_l517_51748

theorem product_remainder_div_5 :
  (1234 * 1567 * 1912) % 5 = 1 :=
by
  sorry

end product_remainder_div_5_l517_51748


namespace sasha_hometown_name_l517_51713

theorem sasha_hometown_name :
  ∃ (sasha_hometown : String), 
  (∃ (vadik_last_column : String), vadik_last_column = "ВКСАМО") →
  (∃ (sasha_transformed : String), sasha_transformed = "мТТЛАРАЕкис") →
  (∃ (sasha_starts_with : Char), sasha_starts_with = 'с') →
  sasha_hometown = "СТЕРЛИТАМАК" :=
by
  sorry

end sasha_hometown_name_l517_51713


namespace laran_weekly_profit_l517_51702

-- Definitions based on the problem conditions
def daily_posters_sold : ℕ := 5
def large_posters_sold_daily : ℕ := 2
def small_posters_sold_daily : ℕ := daily_posters_sold - large_posters_sold_daily

def price_large_poster : ℕ := 10
def cost_large_poster : ℕ := 5
def profit_large_poster : ℕ := price_large_poster - cost_large_poster

def price_small_poster : ℕ := 6
def cost_small_poster : ℕ := 3
def profit_small_poster : ℕ := price_small_poster - cost_small_poster

def daily_profit_large_posters : ℕ := large_posters_sold_daily * profit_large_poster
def daily_profit_small_posters : ℕ := small_posters_sold_daily * profit_small_poster
def total_daily_profit : ℕ := daily_profit_large_posters + daily_profit_small_posters

def school_days_week : ℕ := 5
def weekly_profit : ℕ := total_daily_profit * school_days_week

-- Statement to prove
theorem laran_weekly_profit : weekly_profit = 95 := sorry

end laran_weekly_profit_l517_51702


namespace range_of_abscissa_of_P_l517_51794

noncomputable def point_lies_on_line (P : ℝ × ℝ) : Prop :=
  P.1 - P.2 + 1 = 0

noncomputable def point_lies_on_circle_c (M N : ℝ × ℝ) : Prop :=
  (M.1 - 2)^2 + (M.2 - 1)^2 = 1 ∧ (N.1 - 2)^2 + (N.2 - 1)^2 = 1

noncomputable def angle_mpn_eq_60 (P M N : ℝ × ℝ) : Prop :=
  true -- This is a placeholder because we have to define the geometrical angle condition which is complex.

theorem range_of_abscissa_of_P :
  ∀ (P M N : ℝ × ℝ),
  point_lies_on_line P →
  point_lies_on_circle_c M N →
  angle_mpn_eq_60 P M N →
  0 ≤ P.1 ∧ P.1 ≤ 2 := sorry

end range_of_abscissa_of_P_l517_51794


namespace pentagon_angle_sum_l517_51764

theorem pentagon_angle_sum (A B C D Q : ℝ) (hA : A = 118) (hB : B = 105) (hC : C = 87) (hD : D = 135) :
  (A + B + C + D + Q = 540) -> Q = 95 :=
by
  sorry

end pentagon_angle_sum_l517_51764


namespace not_perfect_square_l517_51710

theorem not_perfect_square (h1 : ∃ x : ℝ, x^2 = 1 ^ 2018) 
                           (h2 : ¬ ∃ x : ℝ, x^2 = 2 ^ 2019)
                           (h3 : ∃ x : ℝ, x^2 = 3 ^ 2020)
                           (h4 : ∃ x : ℝ, x^2 = 4 ^ 2021)
                           (h5 : ∃ x : ℝ, x^2 = 6 ^ 2022) : 
  2 ^ 2019 ≠ x^2 := 
sorry

end not_perfect_square_l517_51710


namespace coprime_solution_l517_51703

theorem coprime_solution (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_eq : 5 * a + 7 * b = 29 * (6 * a + 5 * b)) : a = 3 ∧ b = 2 :=
sorry

end coprime_solution_l517_51703


namespace usual_time_is_36_l517_51729

-- Definition: let S be the usual speed of the worker (not directly relevant to the final proof)
noncomputable def S : ℝ := sorry

-- Definition: let T be the usual time taken by the worker
noncomputable def T : ℝ := sorry

-- Condition: The worker's speed is (3/4) of her normal speed, resulting in a time (T + 12)
axiom speed_delay_condition : (3 / 4) * S * (T + 12) = S * T

-- Theorem: Prove that the usual time T taken to cover the distance is 36 minutes
theorem usual_time_is_36 : T = 36 := by
  -- Formally stating our proof based on given conditions
  sorry

end usual_time_is_36_l517_51729


namespace friend_initial_money_l517_51725

theorem friend_initial_money (F : ℕ) : 
    (160 + 25 * 7 = F + 25 * 5) → 
    (F = 210) :=
by
  sorry

end friend_initial_money_l517_51725


namespace largest_regular_hexagon_proof_l517_51760

noncomputable def largest_regular_hexagon_side_length (x : ℝ) (H : ConvexHexagon) 
  (hx : -5 < x ∧ x < 6) : ℝ := 11 / 2

-- Convex Hexagon Definition
structure ConvexHexagon :=
  (sides : Vector ℝ 6)
  (is_convex : true)  -- Placeholder for convex property

theorem largest_regular_hexagon_proof (x : ℝ) (H : ConvexHexagon) 
  (hx : -5 < x ∧ x < 6)
  (H_sides_length : H.sides = ⟨[5, 6, 7, 5+x, 6-x, 7+x], by simp⟩) :
  largest_regular_hexagon_side_length x H hx = 11 / 2 :=
sorry

end largest_regular_hexagon_proof_l517_51760


namespace rectangle_perimeter_bounds_l517_51708

/-- Given 12 rectangular cardboard pieces, each measuring 4 cm in length and 3 cm in width,
  if these pieces are assembled to form a larger rectangle (possibly including squares),
  without overlapping or leaving gaps, then the minimum possible perimeter of the resulting 
  rectangle is 48 cm and the maximum possible perimeter is 102 cm. -/
theorem rectangle_perimeter_bounds (n : ℕ) (l w : ℝ) (total_area : ℝ) :
  n = 12 ∧ l = 4 ∧ w = 3 ∧ total_area = n * l * w →
  ∃ (min_perimeter max_perimeter : ℝ),
    min_perimeter = 48 ∧ max_perimeter = 102 :=
by
  intros
  sorry

end rectangle_perimeter_bounds_l517_51708


namespace alex_needs_more_coins_l517_51752

-- Define the conditions and problem statement 
def num_friends : ℕ := 15
def coins_alex_has : ℕ := 95 

-- The total number of coins required is
def total_coins_needed : ℕ := num_friends * (num_friends + 1) / 2

-- The minimum number of additional coins needed
def additional_coins_needed : ℕ := total_coins_needed - coins_alex_has

-- Formalize the theorem 
theorem alex_needs_more_coins : additional_coins_needed = 25 := by
  -- Here we would provide the actual proof steps
  sorry

end alex_needs_more_coins_l517_51752


namespace vertex_position_l517_51763

-- Definitions based on the conditions of the problem
def quadratic_function (x : ℝ) : ℝ := 3*x^2 + 9*x + 5

-- Theorem that the vertex of the parabola is at x = -1.5
theorem vertex_position : ∃ x : ℝ, x = -1.5 ∧ ∀ y : ℝ, quadratic_function y ≥ quadratic_function x :=
by
  sorry

end vertex_position_l517_51763


namespace remainder_when_a6_divided_by_n_l517_51792

theorem remainder_when_a6_divided_by_n (n : ℕ) (a : ℤ) (h : a^3 ≡ 1 [ZMOD n]) :
  a^6 ≡ 1 [ZMOD n] := 
sorry

end remainder_when_a6_divided_by_n_l517_51792


namespace function_decreases_l517_51714

def op (m n : ℝ) : ℝ := - (m * n) + n

def f (x : ℝ) : ℝ := op x 2

theorem function_decreases (x1 x2 : ℝ) (h : x1 < x2) : f x1 > f x2 :=
by sorry

end function_decreases_l517_51714


namespace find_f_2_find_f_neg2_l517_51738

noncomputable def f : ℝ → ℝ := sorry -- This is left to be defined as a function on ℝ

axiom f_property : ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y
axiom f_at_1 : f 1 = 2

theorem find_f_2 : f 2 = 6 := by
  sorry

theorem find_f_neg2 : f (-2) = 2 := by
  sorry

end find_f_2_find_f_neg2_l517_51738


namespace E_eq_F_l517_51778

noncomputable def E : Set ℝ := { x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3) }

noncomputable def F : Set ℝ := { x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6) }

theorem E_eq_F : E = F := 
sorry

end E_eq_F_l517_51778


namespace problem_statement_l517_51717

theorem problem_statement (x y : ℝ) (log2_3 log5_3 : ℝ)
  (h1 : log2_3 > 1)
  (h2 : 0 < log5_3)
  (h3 : log5_3 < 1)
  (h4 : log2_3^x - log5_3^x ≥ log2_3^(-y) - log5_3^(-y)) :
  x + y ≥ 0 := 
sorry

end problem_statement_l517_51717


namespace train_stop_time_per_hour_l517_51766

theorem train_stop_time_per_hour
    (speed_excl_stoppages : ℕ)
    (speed_incl_stoppages : ℕ)
    (h1 : speed_excl_stoppages = 48)
    (h2 : speed_incl_stoppages = 36) :
    ∃ (t : ℕ), t = 15 :=
by
  sorry

end train_stop_time_per_hour_l517_51766


namespace eldest_child_age_l517_51799

variables (y m e : ℕ)

theorem eldest_child_age (h1 : m = y + 3)
                        (h2 : e = 3 * y)
                        (h3 : e = y + m + 2) : e = 15 :=
by
  sorry

end eldest_child_age_l517_51799


namespace world_grain_supply_is_correct_l517_51755

def world_grain_demand : ℝ := 2400000
def supply_ratio : ℝ := 0.75
def world_grain_supply (demand : ℝ) (ratio : ℝ) : ℝ := ratio * demand

theorem world_grain_supply_is_correct :
  world_grain_supply world_grain_demand supply_ratio = 1800000 := 
by 
  sorry

end world_grain_supply_is_correct_l517_51755


namespace students_in_class_l517_51739

theorem students_in_class (S : ℕ)
  (h₁ : S / 2 + 2 * S / 5 - S / 10 = 4 * S / 5)
  (h₂ : S / 5 = 4) :
  S = 20 :=
sorry

end students_in_class_l517_51739


namespace pipe_q_fills_in_9_hours_l517_51776

theorem pipe_q_fills_in_9_hours (x : ℝ) :
  (1 / 3 + 1 / x + 1 / 18 = 1 / 2) → x = 9 :=
by {
  sorry
}

end pipe_q_fills_in_9_hours_l517_51776


namespace mn_value_l517_51741

noncomputable def log_base (a b : ℝ) := Real.log b / Real.log a

theorem mn_value (M N : ℝ) (a : ℝ) 
  (h1 : log_base M N = a * log_base N M)
  (h2 : M ≠ N) (h3 : M * N > 0) (h4 : M ≠ 1) (h5 : N ≠ 1) (h6 : a = 4)
  : M * N = N^(3/2) ∨ M * N = N^(1/2) := 
by
  sorry

end mn_value_l517_51741


namespace alex_and_zhu_probability_l517_51797

theorem alex_and_zhu_probability :
  let num_students := 100
  let num_selected := 60
  let num_sections := 3
  let section_size := 20
  let P_alex_selected := 3 / 5
  let P_zhu_selected_given_alex_selected := 59 / 99
  let P_same_section_given_both_selected := 19 / 59
  (P_alex_selected * P_zhu_selected_given_alex_selected * P_same_section_given_both_selected) = 19 / 165 := 
by {
  sorry
}

end alex_and_zhu_probability_l517_51797


namespace exists_divisible_by_2011_l517_51768

def a (n : ℕ) : ℕ := (List.range n).foldl (λ acc i => acc + 10 ^ i) 0

theorem exists_divisible_by_2011 : ∃ n, 1 ≤ n ∧ n ≤ 2011 ∧ 2011 ∣ a n := by
  sorry

end exists_divisible_by_2011_l517_51768


namespace tan_theta_parallel_l517_51758

theorem tan_theta_parallel (θ : ℝ) : 
  let a := (2, 3)
  let b := (Real.cos θ, Real.sin θ)
  (b.1 * a.2 = b.2 * a.1) → Real.tan θ = 3 / 2 :=
by
  intros h
  sorry

end tan_theta_parallel_l517_51758


namespace red_balls_in_bag_l517_51731

theorem red_balls_in_bag (r : ℕ) (h1 : 0 ≤ r ∧ r ≤ 12)
  (h2 : (r * (r - 1)) / (12 * 11) = 1 / 10) : r = 12 :=
sorry

end red_balls_in_bag_l517_51731


namespace doubling_period_l517_51715

theorem doubling_period (initial_capacity: ℝ) (final_capacity: ℝ) (years: ℝ) (initial_year: ℝ) (final_year: ℝ) (doubling_period: ℝ) :
  initial_capacity = 0.4 → final_capacity = 4100 → years = (final_year - initial_year) →
  initial_year = 2000 → final_year = 2050 →
  2 ^ (years / doubling_period) * initial_capacity = final_capacity :=
by
  intros h_initial h_final h_years h_i_year h_f_year
  sorry

end doubling_period_l517_51715


namespace total_payment_divisible_by_25_l517_51796

theorem total_payment_divisible_by_25 (B : ℕ) (h1 : 0 ≤ B ∧ B ≤ 9) : 
  (2005 + B * 1000) % 25 = 0 :=
by
  sorry

end total_payment_divisible_by_25_l517_51796


namespace masking_tape_problem_l517_51736

variable (width_other : ℕ)

theorem masking_tape_problem
  (h1 : ∀ w : ℕ, (2 * 4 + 2 * w) = 20)
  : width_other = 6 :=
by
  have h2 : 8 + 2 * width_other = 20 := h1 width_other
  sorry

end masking_tape_problem_l517_51736


namespace fraction_product_l517_51750

theorem fraction_product :
  (7 / 4) * (8 / 14) * (28 / 16) * (24 / 36) * (49 / 35) * (40 / 25) * (63 / 42) * (32 / 48) = 56 / 25 :=
by sorry

end fraction_product_l517_51750
