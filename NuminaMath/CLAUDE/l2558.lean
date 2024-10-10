import Mathlib

namespace range_of_f_l2558_255875

noncomputable def f (x : ℝ) : ℝ := (3 * x + 4) / (x - 5)

theorem range_of_f :
  Set.range f = {y : ℝ | y < 3 ∨ y > 3} := by sorry

end range_of_f_l2558_255875


namespace wuyang_cup_result_l2558_255831

-- Define the teams
inductive Team : Type
| A : Team
| B : Team
| C : Team
| D : Team

-- Define the positions
inductive Position : Type
| Champion : Position
| RunnerUp : Position
| Third : Position
| Last : Position

-- Define the result type
def Result := Team → Position

-- Define the predictor type
inductive Predictor : Type
| Jia : Predictor
| Yi : Predictor
| Bing : Predictor

-- Define the prediction type
def Prediction := Predictor → Team → Position

-- Define the correctness of a prediction
def is_correct (pred : Prediction) (result : Result) (p : Predictor) (t : Team) : Prop :=
  pred p t = result t

-- Define the condition that each predictor is half right and half wrong
def half_correct (pred : Prediction) (result : Result) (p : Predictor) : Prop :=
  (∃ t1 t2 : Team, t1 ≠ t2 ∧ is_correct pred result p t1 ∧ is_correct pred result p t2) ∧
  (∃ t3 t4 : Team, t3 ≠ t4 ∧ ¬is_correct pred result p t3 ∧ ¬is_correct pred result p t4)

-- Define the predictions
def predictions (pred : Prediction) : Prop :=
  pred Predictor.Jia Team.C = Position.RunnerUp ∧
  pred Predictor.Jia Team.D = Position.Third ∧
  pred Predictor.Yi Team.D = Position.Last ∧
  pred Predictor.Yi Team.A = Position.RunnerUp ∧
  pred Predictor.Bing Team.C = Position.Champion ∧
  pred Predictor.Bing Team.B = Position.RunnerUp

-- State the theorem
theorem wuyang_cup_result :
  ∀ (pred : Prediction) (result : Result),
    predictions pred →
    (∀ p : Predictor, half_correct pred result p) →
    result Team.C = Position.Champion ∧
    result Team.A = Position.RunnerUp ∧
    result Team.D = Position.Third ∧
    result Team.B = Position.Last :=
sorry

end wuyang_cup_result_l2558_255831


namespace possible_values_of_u_l2558_255809

theorem possible_values_of_u (u v : ℝ) (hu : u ≠ 0) (hv : v ≠ 0)
  (eq1 : u + 1/v = 8) (eq2 : v + 1/u = 16/3) :
  u = 4 + Real.sqrt 232 / 4 ∨ u = 4 - Real.sqrt 232 / 4 := by
  sorry

end possible_values_of_u_l2558_255809


namespace closest_point_proof_l2558_255839

-- Define the cheese location
def cheese : ℝ × ℝ := (10, 10)

-- Define the mouse's path
def mouse_path (x : ℝ) : ℝ := -4 * x + 16

-- Define the point of closest approach
def closest_point : ℝ × ℝ := (2, 8)

-- Theorem statement
theorem closest_point_proof :
  -- The closest point is on the mouse's path
  mouse_path closest_point.1 = closest_point.2 ∧
  -- The closest point is indeed the closest to the cheese
  ∀ x : ℝ, x ≠ closest_point.1 →
    (x - cheese.1)^2 + (mouse_path x - cheese.2)^2 >
    (closest_point.1 - cheese.1)^2 + (closest_point.2 - cheese.2)^2 ∧
  -- The sum of coordinates of the closest point is 10
  closest_point.1 + closest_point.2 = 10 :=
sorry

end closest_point_proof_l2558_255839


namespace candy_division_theorem_l2558_255868

/-- Represents the share of candy each person takes -/
structure CandyShare where
  al : Rat
  bert : Rat
  carl : Rat
  dana : Rat

/-- The function that calculates the remaining candy fraction -/
def remainingCandy (shares : CandyShare) : Rat :=
  1 - (shares.al + shares.bert + shares.carl + shares.dana)

/-- The theorem stating the correct remaining candy fraction -/
theorem candy_division_theorem (x : Rat) (shares : CandyShare) :
  shares.al = 3/7 ∧
  shares.bert = 2/7 * (1 - 3/7) ∧
  shares.carl = 1/7 * (1 - 3/7 - 2/7 * (1 - 3/7)) ∧
  shares.dana = 1/7 * (1 - 3/7 - 2/7 * (1 - 3/7) - 1/7 * (1 - 3/7 - 2/7 * (1 - 3/7))) →
  remainingCandy shares = 584/2401 := by
  sorry

#check candy_division_theorem

end candy_division_theorem_l2558_255868


namespace arithmetic_sequence_sum_l2558_255818

/-- 
Given three consecutive terms of an arithmetic sequence with common difference 6,
prove that if their sum is 342, then the terms are 108, 114, and 120.
-/
theorem arithmetic_sequence_sum (a b c : ℕ) : 
  (b = a + 6 ∧ c = b + 6) →  -- consecutive terms with common difference 6
  (a + b + c = 342) →        -- sum is 342
  (a = 108 ∧ b = 114 ∧ c = 120) := by
sorry

end arithmetic_sequence_sum_l2558_255818


namespace x_minus_p_equals_3_minus_2p_l2558_255886

theorem x_minus_p_equals_3_minus_2p (x p : ℝ) (h1 : |x - 3| = p) (h2 : x < 3) :
  x - p = 3 - 2*p := by
  sorry

end x_minus_p_equals_3_minus_2p_l2558_255886


namespace inscribed_square_area_l2558_255808

/-- The area of a square inscribed in the ellipse x²/4 + y²/8 = 1, 
    with its sides parallel to the coordinate axes, is 32/3 -/
theorem inscribed_square_area (x y : ℝ) :
  (∃ s : ℝ, s > 0 ∧ 
    (x^2 / 4 + y^2 / 8 = 1) ∧ 
    (x = s ∨ x = -s) ∧ 
    (y = s ∨ y = -s)) →
  (4 * s^2 = 32 / 3) :=
by sorry

end inscribed_square_area_l2558_255808


namespace expression_equality_l2558_255898

theorem expression_equality : 
  (|(-1)|^2023 : ℝ) + (Real.sqrt 3)^2 - 2 * Real.sin (π / 6) + (1 / 2)⁻¹ = 5 := by
  sorry

end expression_equality_l2558_255898


namespace no_natural_product_l2558_255822

theorem no_natural_product (n : ℕ) : ¬∃ (a b : ℕ), 3 * n + 1 = a * b := by
  sorry

end no_natural_product_l2558_255822


namespace sum_of_digits_of_seven_to_eleven_l2558_255841

theorem sum_of_digits_of_seven_to_eleven (n : ℕ) : 
  (3 + 4)^11 % 100 = 43 → 
  (((3 + 4)^11 / 10) % 10 + (3 + 4)^11 % 10) = 7 :=
by sorry

end sum_of_digits_of_seven_to_eleven_l2558_255841


namespace arc_length_quarter_circle_l2558_255805

/-- Given a circle D with circumference 72 feet and an arc EF subtended by a central angle of 90°,
    prove that the length of arc EF is 18 feet. -/
theorem arc_length_quarter_circle (D : Real) (EF : Real) :
  D = 72 → -- Circumference of circle D is 72 feet
  EF = D / 4 → -- Arc EF is subtended by a 90° angle (1/4 of the circle)
  EF = 18 := by sorry

end arc_length_quarter_circle_l2558_255805


namespace problem_distribution_l2558_255876

def num_problems : ℕ := 5
def num_friends : ℕ := 12

theorem problem_distribution :
  (num_friends ^ num_problems : ℕ) = 248832 :=
by sorry

end problem_distribution_l2558_255876


namespace sequence_count_16_l2558_255881

/-- Represents the number of valid sequences of length n -/
def validSequences : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | 2 => 2
  | n + 2 => 2 * validSequences n

/-- The problem statement -/
theorem sequence_count_16 : validSequences 16 = 256 := by
  sorry

end sequence_count_16_l2558_255881


namespace triangle_third_side_length_l2558_255825

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (ha : a = 5) 
  (hb : b = 7) 
  (hc : c = 10) : 
  a + b > c ∧ b + c > a ∧ c + a > b := by sorry

end triangle_third_side_length_l2558_255825


namespace bottle_capacity_correct_l2558_255834

/-- The capacity of Madeline's water bottle in ounces -/
def bottle_capacity : ℕ := 12

/-- The number of times Madeline refills her water bottle -/
def refills : ℕ := 7

/-- The amount of water Madeline needs to drink after refills in ounces -/
def remaining_water : ℕ := 16

/-- The total amount of water Madeline wants to drink in a day in ounces -/
def total_water : ℕ := 100

/-- Theorem stating that the bottle capacity is correct given the conditions -/
theorem bottle_capacity_correct :
  bottle_capacity * refills + remaining_water = total_water :=
by sorry

end bottle_capacity_correct_l2558_255834


namespace book_arrangement_count_l2558_255882

/-- Represents the number of ways to arrange books on a shelf. -/
def arrange_books (math_books : ℕ) (english_books : ℕ) (science_books : ℕ) : ℕ :=
  let group_arrangements := 6
  let math_arrangements := Nat.factorial math_books
  let english_arrangements := Nat.factorial english_books
  let science_arrangements := Nat.factorial science_books
  group_arrangements * math_arrangements * english_arrangements * science_arrangements

/-- Theorem stating the number of ways to arrange the books on the shelf. -/
theorem book_arrangement_count :
  arrange_books 4 6 2 = 207360 :=
by sorry

end book_arrangement_count_l2558_255882


namespace triangle_properties_l2558_255828

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The angle C in radians -/
def angle_C (t : Triangle) : ℝ :=
  sorry

/-- The angle B in radians -/
def angle_B (t : Triangle) : ℝ :=
  sorry

theorem triangle_properties (t : Triangle) 
  (ha : t.a = 3) 
  (hb : t.b = 5) 
  (hc : t.c = 7) : 
  (angle_C t = 2 * Real.pi / 3) ∧ 
  (Real.sin (angle_B t + Real.pi / 3) = 4 * Real.sqrt 3 / 7) :=
by sorry

end triangle_properties_l2558_255828


namespace lego_castle_ratio_l2558_255889

/-- Proves that the ratio of Legos used for the castle to the total number of Legos is 1:2 --/
theorem lego_castle_ratio :
  let total_legos : ℕ := 500
  let legos_put_back : ℕ := 245
  let missing_legos : ℕ := 5
  let castle_legos : ℕ := total_legos - legos_put_back - missing_legos
  (castle_legos : ℚ) / total_legos = 1 / 2 :=
by sorry

end lego_castle_ratio_l2558_255889


namespace inequality_problem_l2558_255884

theorem inequality_problem (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
sorry

end inequality_problem_l2558_255884


namespace janes_flower_bed_area_l2558_255836

/-- A rectangular flower bed with fence posts -/
structure FlowerBed where
  total_posts : ℕ
  post_spacing : ℝ
  long_side_post_ratio : ℕ

/-- Calculate the area of a flower bed given its specifications -/
def flowerBedArea (fb : FlowerBed) : ℝ :=
  let short_side_posts := (fb.total_posts + 4) / (2 * (fb.long_side_post_ratio + 1))
  let long_side_posts := short_side_posts * fb.long_side_post_ratio
  let short_side_length := (short_side_posts - 1) * fb.post_spacing
  let long_side_length := (long_side_posts - 1) * fb.post_spacing
  short_side_length * long_side_length

/-- Theorem: The area of Jane's flower bed is 144 square feet -/
theorem janes_flower_bed_area :
  let fb : FlowerBed := {
    total_posts := 24,
    post_spacing := 3,
    long_side_post_ratio := 3
  }
  flowerBedArea fb = 144 := by sorry

end janes_flower_bed_area_l2558_255836


namespace inequality_proof_equality_condition_l2558_255802

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) : 
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 := by
  sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) : 
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end inequality_proof_equality_condition_l2558_255802


namespace certain_number_proof_l2558_255863

theorem certain_number_proof :
  ∃! x : ℝ, 0.65 * x = (4 / 5 : ℝ) * 25 + 6 :=
by sorry

end certain_number_proof_l2558_255863


namespace total_work_hours_l2558_255840

theorem total_work_hours (hours_per_day : ℕ) (days_worked : ℕ) (total_hours : ℕ) : 
  hours_per_day = 3 → days_worked = 6 → total_hours = hours_per_day * days_worked → total_hours = 18 := by
  sorry

end total_work_hours_l2558_255840


namespace fred_money_left_l2558_255890

/-- Calculates the amount of money Fred has left after spending half his allowance on movies and earning money from washing a car. -/
def money_left (allowance : ℕ) (car_wash_earnings : ℕ) : ℕ :=
  allowance / 2 + car_wash_earnings

/-- Proves that Fred has 14 dollars left given his allowance and car wash earnings. -/
theorem fred_money_left :
  money_left 16 6 = 14 := by
  sorry

end fred_money_left_l2558_255890


namespace expected_votes_for_candidate_a_l2558_255803

theorem expected_votes_for_candidate_a (total_voters : ℕ) 
  (dem_percent : ℝ) (rep_percent : ℝ) (dem_vote_a : ℝ) (rep_vote_a : ℝ) :
  dem_percent = 0.6 →
  rep_percent = 0.4 →
  dem_vote_a = 0.75 →
  rep_vote_a = 0.3 →
  dem_percent + rep_percent = 1 →
  (dem_percent * dem_vote_a + rep_percent * rep_vote_a) * 100 = 57 := by
  sorry

end expected_votes_for_candidate_a_l2558_255803


namespace sqrt_simplification_l2558_255813

theorem sqrt_simplification : (Real.sqrt 2 * Real.sqrt 20) / Real.sqrt 5 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_simplification_l2558_255813


namespace toys_per_rabbit_l2558_255806

def monday_toys : ℕ := 8
def num_rabbits : ℕ := 34

def total_toys : ℕ :=
  monday_toys +  -- Monday
  (3 * monday_toys) +  -- Tuesday
  (2 * 3 * monday_toys) +  -- Wednesday
  monday_toys +  -- Thursday
  (5 * monday_toys) +  -- Friday
  (2 * 3 * monday_toys) / 2  -- Saturday

theorem toys_per_rabbit : total_toys / num_rabbits = 4 := by
  sorry

end toys_per_rabbit_l2558_255806


namespace add_squared_terms_l2558_255858

theorem add_squared_terms (a : ℝ) : a^2 + 3*a^2 = 4*a^2 := by
  sorry

end add_squared_terms_l2558_255858


namespace tenth_power_sum_l2558_255850

theorem tenth_power_sum (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) : a^10 + b^10 = 93 := by
  sorry

end tenth_power_sum_l2558_255850


namespace passengers_scientific_notation_l2558_255896

/-- Represents the number of passengers in millions -/
def passengers : ℝ := 1.446

/-- Represents the scientific notation of the number of passengers -/
def scientific_notation : ℝ := 1.446 * (10 ^ 6)

/-- Theorem stating that the number of passengers in millions 
    is equal to its scientific notation representation -/
theorem passengers_scientific_notation : 
  passengers * 1000000 = scientific_notation := by sorry

end passengers_scientific_notation_l2558_255896


namespace base4_division_l2558_255835

/-- Converts a base 4 number to base 10 -/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 -/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Theorem stating that the quotient of 1012₄ ÷ 12₄ is 23₄ -/
theorem base4_division :
  (base4ToBase10 [2, 1, 0, 1]) / (base4ToBase10 [2, 1]) = base4ToBase10 [3, 2] := by
  sorry

#eval base4ToBase10 [2, 1, 0, 1]  -- Should output 70
#eval base4ToBase10 [2, 1]        -- Should output 6
#eval base4ToBase10 [3, 2]        -- Should output 11
#eval base10ToBase4 23            -- Should output [2, 3]

end base4_division_l2558_255835


namespace profit_percentage_doubling_l2558_255857

theorem profit_percentage_doubling (cost_price : ℝ) (original_selling_price : ℝ) :
  original_selling_price = cost_price * 1.3 →
  let double_price := original_selling_price * 2
  let new_profit_percentage := (double_price - cost_price) / cost_price * 100
  new_profit_percentage = 160 := by
  sorry

end profit_percentage_doubling_l2558_255857


namespace sin_cos_product_tan_plus_sec_second_quadrant_tan_plus_sec_fourth_quadrant_l2558_255877

/-- The angle α with vertex at the origin and initial side on positive x-axis -/
structure Angle (α : ℝ) : Prop where
  vertex_origin : True
  initial_side_positive_x : True

/-- Point P on the terminal side of angle α -/
structure TerminalPoint (α : ℝ) (x y : ℝ) : Prop where
  on_terminal_side : True

/-- The terminal side of angle α lies on the line y = mx -/
structure TerminalLine (α : ℝ) (m : ℝ) : Prop where
  on_line : True

theorem sin_cos_product (α : ℝ) (h : Angle α) (p : TerminalPoint α (-1) 2) :
  Real.sin α * Real.cos α = -2/5 := by sorry

theorem tan_plus_sec_second_quadrant (α : ℝ) (h : Angle α) (l : TerminalLine α (-3)) 
  (q : 0 < α ∧ α < π) :
  Real.tan α + 3 / Real.cos α = -3 - 3 * Real.sqrt 10 := by sorry

theorem tan_plus_sec_fourth_quadrant (α : ℝ) (h : Angle α) (l : TerminalLine α (-3)) 
  (q : -π/2 < α ∧ α < 0) :
  Real.tan α + 3 / Real.cos α = -3 + 3 * Real.sqrt 10 := by sorry

end sin_cos_product_tan_plus_sec_second_quadrant_tan_plus_sec_fourth_quadrant_l2558_255877


namespace coffee_mixture_cost_l2558_255888

theorem coffee_mixture_cost (cost_A : ℝ) (cost_mixture : ℝ) (total_weight : ℝ) (weight_A : ℝ) (weight_B : ℝ) :
  cost_A = 10 →
  cost_mixture = 11 →
  total_weight = 480 →
  weight_A = 240 →
  weight_B = 240 →
  (total_weight * cost_mixture - weight_A * cost_A) / weight_B = 12 :=
by sorry

end coffee_mixture_cost_l2558_255888


namespace right_triangle_semicircle_segments_l2558_255849

theorem right_triangle_semicircle_segments 
  (a b : ℝ) 
  (ha : a = 75) 
  (hb : b = 100) : 
  ∃ (x y : ℝ), 
    x = 48 ∧ 
    y = 36 ∧ 
    x * (a^2 + b^2) = a * b^2 ∧ 
    y * (a^2 + b^2) = b * a^2 := by
  sorry

end right_triangle_semicircle_segments_l2558_255849


namespace poster_area_l2558_255895

theorem poster_area (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : (3 * x + 4) * (y + 3) = 63) : x * y = 15 := by
  sorry

end poster_area_l2558_255895


namespace arcade_candy_cost_l2558_255865

theorem arcade_candy_cost (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candies : ℕ) :
  whack_a_mole_tickets = 26 →
  skee_ball_tickets = 19 →
  candies = 5 →
  (whack_a_mole_tickets + skee_ball_tickets) / candies = 9 :=
by sorry

end arcade_candy_cost_l2558_255865


namespace starting_lineup_combinations_l2558_255892

def team_size : ℕ := 15
def lineup_size : ℕ := 6
def guaranteed_players : ℕ := 2

theorem starting_lineup_combinations :
  Nat.choose (team_size - guaranteed_players) (lineup_size - guaranteed_players) = 715 := by
  sorry

end starting_lineup_combinations_l2558_255892


namespace probability_two_ones_eight_dice_l2558_255899

/-- The probability of exactly two dice showing a 1 when rolling eight standard 6-sided dice -/
def probability_two_ones (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  ↑(Nat.choose n k) * p^k * (1 - p)^(n - k)

/-- Theorem stating the probability of exactly two ones when rolling eight 6-sided dice -/
theorem probability_two_ones_eight_dice :
  probability_two_ones 8 2 (1/6) = 28 * (1/6)^2 * (5/6)^6 :=
sorry

end probability_two_ones_eight_dice_l2558_255899


namespace final_salary_matches_expected_l2558_255867

/-- Calculates the final take-home salary after a raise, pay cut, and tax --/
def finalSalary (initialSalary : ℝ) (raisePercent : ℝ) (cutPercent : ℝ) (taxPercent : ℝ) : ℝ :=
  let salaryAfterRaise := initialSalary * (1 + raisePercent)
  let salaryAfterCut := salaryAfterRaise * (1 - cutPercent)
  salaryAfterCut * (1 - taxPercent)

/-- Theorem stating that the final salary matches the expected value --/
theorem final_salary_matches_expected :
  finalSalary 2500 0.25 0.15 0.10 = 2390.63 := by
  sorry

#eval finalSalary 2500 0.25 0.15 0.10

end final_salary_matches_expected_l2558_255867


namespace intersection_A_B_when_a_is_one_range_of_a_when_B_subset_complementA_l2558_255847

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | (a * x - 1) * (x + 2) ≥ 0}

-- Define the complement of A
def complementA : Set ℝ := {x | x ≤ -1 ∨ x ≥ 2}

-- Theorem for part (1)
theorem intersection_A_B_when_a_is_one :
  A ∩ B 1 = {x | 1 ≤ x ∧ x < 2} := by sorry

-- Theorem for part (2)
theorem range_of_a_when_B_subset_complementA :
  ∀ a > 0, (B a ⊆ complementA) ↔ (0 < a ∧ a ≤ 1/2) := by sorry

end intersection_A_B_when_a_is_one_range_of_a_when_B_subset_complementA_l2558_255847


namespace thompson_exam_rule_l2558_255887

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (answered_all_correctly : Student → Prop)
variable (received_C_or_higher : Student → Prop)

-- State the theorem
theorem thompson_exam_rule 
  (h : ∀ s : Student, ¬(answered_all_correctly s) → ¬(received_C_or_higher s)) :
  ∀ s : Student, received_C_or_higher s → answered_all_correctly s :=
by sorry

end thompson_exam_rule_l2558_255887


namespace odd_prime_square_difference_l2558_255812

theorem odd_prime_square_difference (d : ℕ) : 
  Nat.Prime d → 
  d % 2 = 1 → 
  ∃ m : ℕ, 89 - (d + 3)^2 = m^2 → 
  d = 5 := by sorry

end odd_prime_square_difference_l2558_255812


namespace henrys_age_l2558_255880

theorem henrys_age (h s : ℕ) : 
  h + 8 = 3 * (s - 1) →
  (h - 25) + (s - 25) = 83 →
  h = 97 :=
by sorry

end henrys_age_l2558_255880


namespace max_sides_convex_polygon_l2558_255860

/-- The maximum number of sides in a convex polygon with interior angles in arithmetic sequence -/
theorem max_sides_convex_polygon (n : ℕ) : n ≤ 8 :=
  let interior_angle (k : ℕ) := 100 + 10 * (k - 1)
  have h1 : ∀ k, k ≤ n → interior_angle k < 180 := by sorry
  have h2 : ∀ k, 1 ≤ k → k ≤ n → 0 < interior_angle k := by sorry
  have h3 : (n - 2) * 180 = (interior_angle 1 + interior_angle n) * n / 2 := by sorry
sorry

#check max_sides_convex_polygon

end max_sides_convex_polygon_l2558_255860


namespace sequence_properties_l2558_255843

/-- Definition of sequence a_n -/
def a (n : ℕ) : ℝ := sorry

/-- Definition of S_n as the sum of first n terms of a_n -/
def S (n : ℕ) : ℝ := sorry

/-- Definition of sequence b_n -/
def b (n : ℕ) : ℝ := sorry

/-- Definition of sequence c_n -/
def c (n : ℕ) : ℝ := a n * b n

/-- Definition of T_n as the sum of first n terms of c_n -/
def T (n : ℕ) : ℝ := sorry

/-- Main theorem -/
theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n = (S n + 2) / 2) ∧ 
  (b 1 = 1) ∧
  (∀ n : ℕ, n ≥ 1 → b n - b (n + 1) + 2 = 0) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 2^n) ∧
  (∀ n : ℕ, n ≥ 1 → b n = 2*n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → T n = (2*n - 3) * 2^(n+1) + 6) := by
  sorry

end sequence_properties_l2558_255843


namespace oranges_in_bin_l2558_255848

theorem oranges_in_bin (initial : ℕ) (thrown_away : ℕ) (added : ℕ) :
  initial ≥ thrown_away →
  initial - thrown_away + added = initial + added - thrown_away :=
by sorry

end oranges_in_bin_l2558_255848


namespace petya_stickers_l2558_255800

/-- Calculates the final number of stickers after a series of trades -/
def final_stickers (initial : ℕ) (trade_in : ℕ) (trade_out : ℕ) (num_trades : ℕ) : ℕ :=
  initial + num_trades * (trade_out - trade_in)

/-- Theorem: Petya will have 121 stickers after 30 trades -/
theorem petya_stickers :
  final_stickers 1 1 5 30 = 121 := by
  sorry

end petya_stickers_l2558_255800


namespace solution_in_interval_l2558_255864

theorem solution_in_interval : ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2^x + x - 2 = 0 := by
  sorry

end solution_in_interval_l2558_255864


namespace tshirt_count_l2558_255856

/-- The price of a pant in rupees -/
def pant_price : ℝ := sorry

/-- The price of a t-shirt in rupees -/
def tshirt_price : ℝ := sorry

/-- The total cost of 3 pants and 6 t-shirts in rupees -/
def total_cost_1 : ℝ := 750

/-- The total cost of 1 pant and 12 t-shirts in rupees -/
def total_cost_2 : ℝ := 750

/-- The amount to be spent on t-shirts in rupees -/
def tshirt_budget : ℝ := 400

theorem tshirt_count : 
  3 * pant_price + 6 * tshirt_price = total_cost_1 →
  pant_price + 12 * tshirt_price = total_cost_2 →
  (tshirt_budget / tshirt_price : ℝ) = 8 := by
sorry

end tshirt_count_l2558_255856


namespace smallest_x_for_cube_equation_l2558_255879

theorem smallest_x_for_cube_equation (N : ℕ+) (h : 1260 * x = N^3) : 
  ∃ (x : ℕ), x = 7350 ∧ ∀ (y : ℕ), 1260 * y = N^3 → x ≤ y := by
  sorry

end smallest_x_for_cube_equation_l2558_255879


namespace zero_exponent_rule_l2558_255833

theorem zero_exponent_rule (a b : ℤ) (hb : b ≠ 0) : (a / b : ℚ) ^ (0 : ℕ) = 1 := by
  sorry

end zero_exponent_rule_l2558_255833


namespace product_of_positive_real_solutions_l2558_255897

theorem product_of_positive_real_solutions : ∃ (S : Finset (ℂ)),
  (∀ z ∈ S, z^8 = -256 ∧ z.re > 0) ∧
  (∀ z : ℂ, z^8 = -256 ∧ z.re > 0 → z ∈ S) ∧
  (S.prod id = 8) :=
sorry

end product_of_positive_real_solutions_l2558_255897


namespace modulus_of_complex_number_l2558_255873

theorem modulus_of_complex_number : 
  let z : ℂ := Complex.I * (1 + 2 * Complex.I)
  Complex.abs z = Real.sqrt 5 := by sorry

end modulus_of_complex_number_l2558_255873


namespace knights_probability_l2558_255846

/-- The number of knights seated at the round table -/
def total_knights : ℕ := 30

/-- The number of knights chosen randomly -/
def chosen_knights : ℕ := 4

/-- The probability that at least two of the chosen knights were sitting next to each other -/
def Q : ℚ := 4456 / 4701

theorem knights_probability :
  Q = 1 - (total_knights * (total_knights - 4) * (total_knights - 8) * (total_knights - 12)) /
      (total_knights * (total_knights - 1) * (total_knights - 2) * (total_knights - 3)) :=
sorry

end knights_probability_l2558_255846


namespace probability_is_three_fifths_l2558_255827

/-- The set of letters in the word "STATISTICS" -/
def statistics_letters : Finset Char := {'S', 'T', 'A', 'I', 'C'}

/-- The set of letters in the word "TEST" -/
def test_letters : Finset Char := {'T', 'E', 'S'}

/-- The number of occurrences of each letter in "STATISTICS" -/
def letter_count (c : Char) : ℕ :=
  if c = 'S' then 3
  else if c = 'T' then 3
  else if c = 'A' then 1
  else if c = 'I' then 2
  else if c = 'C' then 1
  else 0

/-- The total number of tiles -/
def total_tiles : ℕ := statistics_letters.sum letter_count

/-- The number of tiles with letters from "TEST" -/
def test_tiles : ℕ := (statistics_letters ∩ test_letters).sum letter_count

/-- The probability of selecting a tile with a letter from "TEST" -/
def probability : ℚ := test_tiles / total_tiles

theorem probability_is_three_fifths : probability = 3 / 5 := by
  sorry


end probability_is_three_fifths_l2558_255827


namespace cart_distance_theorem_l2558_255814

/-- Represents a cart with two wheels -/
structure Cart where
  front_wheel_circumference : ℝ
  back_wheel_circumference : ℝ

/-- Calculates the distance traveled by the cart -/
def distance_traveled (c : Cart) (back_wheel_revolutions : ℝ) : ℝ :=
  c.back_wheel_circumference * back_wheel_revolutions

theorem cart_distance_theorem (c : Cart) (back_wheel_revolutions : ℝ) :
  c.front_wheel_circumference = 30 →
  c.back_wheel_circumference = 33 →
  c.front_wheel_circumference * (back_wheel_revolutions + 5) = c.back_wheel_circumference * back_wheel_revolutions →
  distance_traveled c back_wheel_revolutions = 1650 := by
  sorry

#check cart_distance_theorem

end cart_distance_theorem_l2558_255814


namespace line_y_intercept_l2558_255874

/-- A line with slope -3 and x-intercept (4, 0) has y-intercept (0, 12) -/
theorem line_y_intercept (line : ℝ → ℝ) (slope : ℝ) (x_intercept : ℝ × ℝ) :
  slope = -3 →
  x_intercept = (4, 0) →
  (∀ x, line x = slope * x + line 0) →
  line 4 = 0 →
  line 0 = 12 := by
sorry

end line_y_intercept_l2558_255874


namespace class_gender_ratio_l2558_255821

theorem class_gender_ratio :
  ∀ (girls boys : ℕ),
  girls = boys + 6 →
  girls + boys = 36 →
  (girls : ℚ) / (boys : ℚ) = 7 / 5 := by
  sorry

end class_gender_ratio_l2558_255821


namespace abs_neg_one_third_l2558_255891

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by sorry

end abs_neg_one_third_l2558_255891


namespace quadratic_inequality_solution_l2558_255826

-- Define the quadratic function
def f (x : ℝ) := 3 * x^2 - 7 * x - 6

-- Define the solution set
def solution_set : Set ℝ := {x | -2/3 < x ∧ x < 3}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = solution_set :=
sorry

end quadratic_inequality_solution_l2558_255826


namespace unique_star_solution_l2558_255870

/-- Definition of the ⋆ operation -/
def star (x y : ℝ) : ℝ := 5*x - 4*y + 2*x*y

/-- Theorem stating that there exists exactly one real number y such that 4 ⋆ y = 20 -/
theorem unique_star_solution : ∃! y : ℝ, star 4 y = 20 := by
  sorry

end unique_star_solution_l2558_255870


namespace two_envelopes_require_fee_l2558_255885

-- Define the envelope structure
structure Envelope where
  name : String
  length : ℚ
  height : ℚ

-- Define the condition for additional fee
def requiresAdditionalFee (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 1.5 || ratio > 2.8

-- Define the list of envelopes
def envelopes : List Envelope := [
  ⟨"E", 7, 5⟩,
  ⟨"F", 10, 4⟩,
  ⟨"G", 5, 5⟩,
  ⟨"H", 14, 5⟩
]

-- Theorem statement
theorem two_envelopes_require_fee :
  (envelopes.filter requiresAdditionalFee).length = 2 := by
  sorry

end two_envelopes_require_fee_l2558_255885


namespace josh_remaining_money_l2558_255878

/-- Calculates the remaining money after spending two amounts -/
def remaining_money (initial : ℚ) (spent1 : ℚ) (spent2 : ℚ) : ℚ :=
  initial - (spent1 + spent2)

/-- Theorem: Given Josh's initial $9 and his spending of $1.75 and $1.25, he has $6 left -/
theorem josh_remaining_money :
  remaining_money 9 (175/100) (125/100) = 6 := by
  sorry

end josh_remaining_money_l2558_255878


namespace equation_solution_l2558_255859

theorem equation_solution : 
  let x : ℝ := 405 / 8
  (2 * x - 60) / 3 = (2 * x - 5) / 7 := by sorry

end equation_solution_l2558_255859


namespace sqrt_eight_minus_sqrt_two_equals_sqrt_two_l2558_255866

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two : 
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt_eight_minus_sqrt_two_equals_sqrt_two_l2558_255866


namespace joan_book_revenue_l2558_255845

def total_revenue (total_books : ℕ) (books_at_4 : ℕ) (books_at_7 : ℕ) (price_4 : ℕ) (price_7 : ℕ) (price_10 : ℕ) : ℕ :=
  let remaining_books := total_books - books_at_4 - books_at_7
  books_at_4 * price_4 + books_at_7 * price_7 + remaining_books * price_10

theorem joan_book_revenue :
  total_revenue 33 15 6 4 7 10 = 222 := by
  sorry

end joan_book_revenue_l2558_255845


namespace rectangle_arrangement_probability_l2558_255823

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square -/
structure Square where
  side_length : ℕ

/-- Represents a line segment connecting midpoints of opposite sides of a square -/
structure MidpointLine where
  square : Square

/-- Represents an arrangement of rectangles in a square -/
structure Arrangement where
  square : Square
  rectangles : List Rectangle

/-- Checks if an arrangement is valid (no overlapping rectangles) -/
def is_valid_arrangement (arr : Arrangement) : Prop := sorry

/-- Checks if an arrangement crosses the midpoint line -/
def crosses_midpoint_line (arr : Arrangement) (line : MidpointLine) : Prop := sorry

/-- Counts the number of valid arrangements -/
def count_valid_arrangements (square : Square) (rect_type : Rectangle) (num_rect : ℕ) : ℕ := sorry

/-- Counts the number of valid arrangements that don't cross the midpoint line -/
def count_non_crossing_arrangements (square : Square) (rect_type : Rectangle) (num_rect : ℕ) (line : MidpointLine) : ℕ := sorry

theorem rectangle_arrangement_probability :
  let square := Square.mk 4
  let rect_type := Rectangle.mk 1 2
  let num_rect := 8
  let line := MidpointLine.mk square
  let total_arrangements := count_valid_arrangements square rect_type num_rect
  let non_crossing_arrangements := count_non_crossing_arrangements square rect_type num_rect line
  (non_crossing_arrangements : ℚ) / total_arrangements = 25 / 36 := by sorry

end rectangle_arrangement_probability_l2558_255823


namespace find_P_value_l2558_255869

theorem find_P_value (P Q R B C y z : ℝ) 
  (eq1 : P = Q + R + 32)
  (eq2 : y = B + C + P + z)
  (eq3 : z = Q - R)
  (eq4 : B = 1/3 * P)
  (eq5 : C = 1/3 * P) :
  P = 64 := by
  sorry

end find_P_value_l2558_255869


namespace base_flavors_count_l2558_255844

/-- The number of variations for each base flavor of pizza -/
def variations : ℕ := 4

/-- The total number of pizza varieties available -/
def total_varieties : ℕ := 16

/-- The number of base flavors of pizza -/
def base_flavors : ℕ := total_varieties / variations

theorem base_flavors_count : base_flavors = 4 := by
  sorry

end base_flavors_count_l2558_255844


namespace cube_volume_from_surface_area_l2558_255851

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 864 → volume = 1728 → 
  (∃ (side_length : ℝ), 
    surface_area = 6 * side_length^2 ∧ 
    volume = side_length^3) :=
by
  sorry

end cube_volume_from_surface_area_l2558_255851


namespace arithmetic_sqrt_of_nine_l2558_255861

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

-- State the theorem
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 := by
  sorry

end arithmetic_sqrt_of_nine_l2558_255861


namespace tenth_line_correct_l2558_255830

def ninthLine : String := "311311222113111231131112322211231231131112"

def countConsecutive (s : String) : String :=
  sorry

theorem tenth_line_correct : 
  countConsecutive ninthLine = "13211321322111312211" := by
  sorry

end tenth_line_correct_l2558_255830


namespace max_roses_for_680_l2558_255801

/-- Represents the price of roses for different quantities -/
structure RosePrices where
  individual : ℝ
  dozen : ℝ
  two_dozen : ℝ

/-- Calculates the maximum number of roses that can be purchased given a budget and price structure -/
def max_roses (budget : ℝ) (prices : RosePrices) : ℕ :=
  sorry

/-- The theorem stating the maximum number of roses that can be purchased with $680 -/
theorem max_roses_for_680 :
  let prices : RosePrices := {
    individual := 4.5,
    dozen := 36,
    two_dozen := 50
  }
  max_roses 680 prices = 318 := by
  sorry

end max_roses_for_680_l2558_255801


namespace no_winning_strategy_l2558_255817

/-- Represents a strategy for deciding when to stop in the card game. -/
def Strategy : Type := List Bool → Bool

/-- Represents the state of the game at any point. -/
structure GameState :=
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- The initial state of the game with a standard deck. -/
def initial_state : GameState :=
  { red_cards := 26, black_cards := 26 }

/-- Calculates the probability of winning given a game state and a strategy. -/
def winning_probability (state : GameState) (strategy : Strategy) : ℚ :=
  sorry

/-- Theorem stating that no strategy can have a winning probability greater than 0.5. -/
theorem no_winning_strategy :
  ∀ (strategy : Strategy),
    winning_probability initial_state strategy ≤ 1/2 :=
sorry

end no_winning_strategy_l2558_255817


namespace unique_quadratic_root_condition_l2558_255842

theorem unique_quadratic_root_condition (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + (b + 1/b) * x + c = 0) ↔ 
  c = 3/2 := by
sorry

end unique_quadratic_root_condition_l2558_255842


namespace payment_calculation_l2558_255816

theorem payment_calculation (payment_rate : ℚ) (rooms_cleaned : ℚ) :
  payment_rate = 13 / 3 →
  rooms_cleaned = 8 / 5 →
  payment_rate * rooms_cleaned = 104 / 15 := by
  sorry

end payment_calculation_l2558_255816


namespace algebraic_expression_simplification_and_evaluation_l2558_255855

theorem algebraic_expression_simplification_and_evaluation :
  let x : ℝ := 4 * Real.sin (45 * π / 180) - 2
  let original_expression := (1 / (x - 1)) / ((x + 2) / (x^2 - 2*x + 1)) - x / (x + 2)
  let simplified_expression := -1 / (x + 2)
  original_expression = simplified_expression ∧ simplified_expression = -Real.sqrt 2 / 4 := by
  sorry

end algebraic_expression_simplification_and_evaluation_l2558_255855


namespace class_average_calculation_l2558_255832

theorem class_average_calculation (total_students : ℕ) (monday_students : ℕ) (tuesday_students : ℕ)
  (monday_average : ℚ) (tuesday_average : ℚ) :
  total_students = 28 →
  monday_students = 24 →
  tuesday_students = 4 →
  monday_average = 82/100 →
  tuesday_average = 90/100 →
  let overall_average := (monday_students * monday_average + tuesday_students * tuesday_average) / total_students
  ∃ ε > 0, |overall_average - 83/100| < ε :=
by sorry

end class_average_calculation_l2558_255832


namespace amount_per_painting_l2558_255804

/-- Hallie's art earnings -/
def total_earnings : ℕ := 300

/-- Number of paintings sold -/
def paintings_sold : ℕ := 3

/-- Theorem: The amount earned per painting is $100 -/
theorem amount_per_painting :
  total_earnings / paintings_sold = 100 := by sorry

end amount_per_painting_l2558_255804


namespace min_value_at_neg_pi_half_l2558_255871

/-- The function f(x) = x + 2cos(x) has its minimum value on the interval [-π/2, 0] at x = -π/2 -/
theorem min_value_at_neg_pi_half :
  let f : ℝ → ℝ := λ x ↦ x + 2 * Real.cos x
  let a : ℝ := -π/2
  let b : ℝ := 0
  ∀ x ∈ Set.Icc a b, f a ≤ f x := by
  sorry

end min_value_at_neg_pi_half_l2558_255871


namespace sunflower_height_in_meters_l2558_255837

-- Define constants
def sister_height_feet : ℝ := 4.15
def sister_additional_height_cm : ℝ := 37
def sunflower_height_difference_inches : ℝ := 63

-- Define conversion factors
def inches_per_foot : ℝ := 12
def cm_per_inch : ℝ := 2.54
def cm_per_meter : ℝ := 100

-- Theorem statement
theorem sunflower_height_in_meters :
  let sister_height_cm := sister_height_feet * inches_per_foot * cm_per_inch + sister_additional_height_cm
  let sunflower_height_cm := sister_height_cm + sunflower_height_difference_inches * cm_per_inch
  sunflower_height_cm / cm_per_meter = 3.23512 := by
  sorry

end sunflower_height_in_meters_l2558_255837


namespace points_lost_in_last_round_l2558_255810

-- Define the variables
def first_round_points : ℕ := 17
def second_round_points : ℕ := 6
def final_points : ℕ := 7

-- Define the theorem
theorem points_lost_in_last_round :
  (first_round_points + second_round_points) - final_points = 16 := by
  sorry

end points_lost_in_last_round_l2558_255810


namespace job_completion_time_l2558_255811

theorem job_completion_time (x : ℝ) (h1 : x > 0) (h2 : 9/x + 4/10 = 1) : x = 15 := by
  sorry

end job_completion_time_l2558_255811


namespace remaining_wire_length_l2558_255807

/-- Given a wire of length 60 cm and a square with side length 9 cm made from this wire,
    the remaining wire length is 24 cm. -/
theorem remaining_wire_length (total_wire : ℝ) (square_side : ℝ) (remaining_wire : ℝ) :
  total_wire = 60 ∧ square_side = 9 →
  remaining_wire = total_wire - 4 * square_side →
  remaining_wire = 24 := by
sorry

end remaining_wire_length_l2558_255807


namespace right_triangle_inequality_l2558_255829

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b < c) 
    (h4 : a^2 + b^2 = c^2) : 
  (1/a + 1/b + 1/c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) ∧ 
  ∃ (a' b' c' : ℝ), 0 < a' ∧ a' ≤ b' ∧ b' < c' ∧ a'^2 + b'^2 = c'^2 ∧
    (1/a' + 1/b' + 1/c') = (5 + 3 * Real.sqrt 2) / (a' + b' + c') := by
  sorry

end right_triangle_inequality_l2558_255829


namespace log_equation_solution_l2558_255872

theorem log_equation_solution :
  ∃! x : ℝ, (Real.log (Real.sqrt (7 * x + 3)) + Real.log (Real.sqrt (4 * x + 5)) = 1 / 2 + Real.log 3) ∧
             (7 * x + 3 > 0) ∧ (4 * x + 5 > 0) := by
  sorry

end log_equation_solution_l2558_255872


namespace max_value_ratio_l2558_255824

theorem max_value_ratio (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0)
  (h5 : a^2 + b^2 + c^2 + d^2 = ((a + b + c + d)^2) / 3) :
  (a + c) / (b + d) ≤ (7 + 2 * Real.sqrt 6) / 5 ∧ 
  ∃ a b c d, a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d > 0 ∧
    a^2 + b^2 + c^2 + d^2 = ((a + b + c + d)^2) / 3 ∧
    (a + c) / (b + d) = (7 + 2 * Real.sqrt 6) / 5 :=
by sorry

end max_value_ratio_l2558_255824


namespace cone_angle_and_ratio_l2558_255838

/-- For a cone with ratio k of total surface area to axial cross-section area,
    prove the angle between height and slant height, and permissible k values. -/
theorem cone_angle_and_ratio (k : ℝ) (α : ℝ) : k > π ∧ α = π/2 - 2 * Real.arctan (π/k) → 
  (π * (Real.sin α + 1)) / Real.cos α = k := by
  sorry

end cone_angle_and_ratio_l2558_255838


namespace stock_price_change_l2558_255820

theorem stock_price_change (total_stocks : ℕ) (higher_price_stocks : ℕ) 
  (h1 : total_stocks = 1980)
  (h2 : higher_price_stocks = (total_stocks - higher_price_stocks) * 6 / 5) :
  higher_price_stocks = 1080 := by
  sorry

end stock_price_change_l2558_255820


namespace computer_profit_percentage_l2558_255894

theorem computer_profit_percentage (cost : ℝ) 
  (h1 : 2240 = cost * 1.4) 
  (h2 : 2400 > cost) : 
  (2400 - cost) / cost = 0.5 := by
sorry

end computer_profit_percentage_l2558_255894


namespace sum_of_cubes_l2558_255815

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 3) : x^3 + y^3 = -10 := by
  sorry

end sum_of_cubes_l2558_255815


namespace bottles_per_player_first_break_l2558_255853

/-- Proves that each player took 2 bottles during the first break of a soccer match --/
theorem bottles_per_player_first_break :
  let total_bottles : ℕ := 4 * 12  -- 4 dozen
  let num_players : ℕ := 11
  let bottles_remaining : ℕ := 15
  let bottles_end_game : ℕ := num_players * 1  -- each player takes 1 bottle at the end

  let bottles_first_break : ℕ := total_bottles - bottles_end_game - bottles_remaining

  (bottles_first_break / num_players : ℚ) = 2 := by
sorry

end bottles_per_player_first_break_l2558_255853


namespace max_min_difference_z_l2558_255893

theorem max_min_difference_z (x y z : ℝ) 
  (sum_eq : x + y + z = 6)
  (sum_squares_eq : x^2 + y^2 + z^2 = 24) :
  ∃ (z_max z_min : ℝ),
    (∀ w, w = x ∨ w = y ∨ w = z → z_min ≤ w ∧ w ≤ z_max) ∧
    z_max - z_min = 4 :=
sorry

end max_min_difference_z_l2558_255893


namespace rectangle_width_l2558_255819

/-- Given a rectangle with perimeter 150 cm and length 15 cm greater than width, prove the width is 30 cm. -/
theorem rectangle_width (w l : ℝ) (h1 : l = w + 15) (h2 : 2 * l + 2 * w = 150) : w = 30 := by
  sorry

end rectangle_width_l2558_255819


namespace function_overlap_with_inverse_l2558_255852

theorem function_overlap_with_inverse (a b c d : ℝ) (h1 : a ≠ 0 ∨ c ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ (a * x + b) / (c * x + d)
  (∀ x, f (f x) = x) →
  ((a + d = 0 ∧ ∃ k, f = λ x ↦ (k * x + b) / (c * x - k)) ∨ f = id) :=
by sorry

end function_overlap_with_inverse_l2558_255852


namespace equidistant_point_y_coordinate_l2558_255862

/-- The y-coordinate of a point on the y-axis equidistant from (5, 0) and (3, 6) is 5/3 -/
theorem equidistant_point_y_coordinate :
  let A : ℝ × ℝ := (5, 0)
  let B : ℝ × ℝ := (3, 6)
  let P : ℝ → ℝ × ℝ := fun y ↦ (0, y)
  ∃ y : ℝ, (dist (P y) A)^2 = (dist (P y) B)^2 ∧ y = 5/3 :=
by sorry


end equidistant_point_y_coordinate_l2558_255862


namespace cost_price_calculation_l2558_255854

/-- Proves that the cost price of an article is 1200, given that it was sold at a 40% profit for 1680. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
    (h1 : selling_price = 1680)
    (h2 : profit_percentage = 40) : 
  selling_price / (1 + profit_percentage / 100) = 1200 := by
  sorry

end cost_price_calculation_l2558_255854


namespace range_of_a_l2558_255883

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) = False → a < -2 ∨ a > 2 := by
  sorry

end range_of_a_l2558_255883
