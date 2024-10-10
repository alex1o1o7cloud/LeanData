import Mathlib

namespace arithmetic_equality_l185_18564

theorem arithmetic_equality : (30 - (3010 - 310)) + (3010 - (310 - 30)) = 60 := by
  sorry

end arithmetic_equality_l185_18564


namespace unique_valid_result_exists_correct_answers_for_71_score_l185_18588

/-- Represents the score and correct answers for a math competition. -/
structure CompetitionResult where
  groupA_correct : Nat
  groupB_correct : Nat
  groupB_incorrect : Nat
  total_score : Int

/-- Checks if the CompetitionResult is valid according to the competition rules. -/
def is_valid_result (r : CompetitionResult) : Prop :=
  r.groupA_correct ≤ 5 ∧
  r.groupB_correct + r.groupB_incorrect ≤ 12 ∧
  r.total_score = 8 * r.groupA_correct + 5 * r.groupB_correct - 2 * r.groupB_incorrect

/-- Theorem stating that there is a unique valid result with a total score of 71 and 13 correct answers. -/
theorem unique_valid_result_exists :
  ∃! r : CompetitionResult,
    is_valid_result r ∧
    r.total_score = 71 ∧
    r.groupA_correct + r.groupB_correct = 13 :=
  sorry

/-- Theorem stating that any valid result with a total score of 71 must have 13 correct answers. -/
theorem correct_answers_for_71_score :
  ∀ r : CompetitionResult,
    is_valid_result r → r.total_score = 71 →
    r.groupA_correct + r.groupB_correct = 13 :=
  sorry

end unique_valid_result_exists_correct_answers_for_71_score_l185_18588


namespace sector_area_120_deg_sqrt3_radius_l185_18579

/-- The area of a circular sector with central angle 120° and radius √3 is π. -/
theorem sector_area_120_deg_sqrt3_radius (π : ℝ) : 
  let angle : ℝ := 120 * π / 180  -- Convert 120° to radians
  let radius : ℝ := Real.sqrt 3
  let sector_area := (1 / 2) * angle * radius^2
  sector_area = π := by sorry

end sector_area_120_deg_sqrt3_radius_l185_18579


namespace attendance_proof_l185_18570

/-- Calculates the total attendance given the number of adults and children -/
def total_attendance (adults : ℕ) (children : ℕ) : ℕ :=
  adults + children

/-- Theorem: The total attendance for 280 adults and 120 children is 400 -/
theorem attendance_proof :
  total_attendance 280 120 = 400 := by
  sorry

end attendance_proof_l185_18570


namespace set_equality_implies_sum_l185_18555

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → 
  a^2004 + b^2005 = 1 := by
  sorry

end set_equality_implies_sum_l185_18555


namespace mischievous_quadratic_min_root_product_l185_18557

/-- A quadratic polynomial with real coefficients and leading coefficient 1 -/
def QuadraticPolynomial (r s : ℝ) (x : ℝ) : ℝ := x^2 - (r + s) * x + r * s

/-- A polynomial is mischievous if p(p(x)) = 0 has exactly four real roots -/
def IsMischievous (p : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x, p (p x) = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)

/-- The theorem stating that the mischievous quadratic polynomial with minimized root product evaluates to 1 at x = 1 -/
theorem mischievous_quadratic_min_root_product (r s : ℝ) :
  IsMischievous (QuadraticPolynomial r s) →
  (∀ r' s' : ℝ, IsMischievous (QuadraticPolynomial r' s') → r * s ≤ r' * s') →
  QuadraticPolynomial r s 1 = 1 := by
  sorry

end mischievous_quadratic_min_root_product_l185_18557


namespace jogging_problem_l185_18514

/-- Jogging problem -/
theorem jogging_problem (total_distance : ℝ) (total_time : ℝ) (halfway_point : ℝ) :
  total_distance = 3 →
  total_time = 24 →
  halfway_point = total_distance / 2 →
  (halfway_point / total_distance) * total_time = 12 :=
by sorry

end jogging_problem_l185_18514


namespace A_initial_investment_l185_18578

/-- Represents the initial investment of A in rupees -/
def A_investment : ℝ := sorry

/-- Represents B's contribution to the capital in rupees -/
def B_investment : ℝ := 16200

/-- Represents the number of months A's investment was active -/
def A_months : ℝ := 12

/-- Represents the number of months B's investment was active -/
def B_months : ℝ := 5

/-- Represents the ratio of A's profit share -/
def A_profit_ratio : ℝ := 2

/-- Represents the ratio of B's profit share -/
def B_profit_ratio : ℝ := 3

theorem A_initial_investment : 
  (A_investment * A_months) / (B_investment * B_months) = A_profit_ratio / B_profit_ratio →
  A_investment = 4500 := by
sorry

end A_initial_investment_l185_18578


namespace line_intersects_circle_l185_18505

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the point P
def point_P : ℝ × ℝ := (3, 0)

-- Define a line passing through point P
def line_through_P (m : ℝ) (x y : ℝ) : Prop := y = m * (x - point_P.1)

-- Theorem statement
theorem line_intersects_circle :
  ∃ (m : ℝ) (x y : ℝ), line_through_P m x y ∧ circle_C x y :=
sorry

end line_intersects_circle_l185_18505


namespace shop_inventory_l185_18566

theorem shop_inventory (large : ℕ) (medium : ℕ) (sold : ℕ) (left : ℕ) (small : ℕ) :
  large = 22 →
  medium = 50 →
  sold = 83 →
  left = 13 →
  large + medium + small = sold + left →
  small = 24 := by
sorry

end shop_inventory_l185_18566


namespace quadratic_solutions_second_eq_solutions_l185_18503

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 + x - 3 = 0

-- Define the second equation
def second_eq (x : ℝ) : Prop := (2*x + 1)^2 = 3*(2*x + 1)

-- Theorem for the first equation
theorem quadratic_solutions :
  ∃ x1 x2 : ℝ, 
    quadratic_eq x1 ∧ 
    quadratic_eq x2 ∧ 
    x1 = (-1 + Real.sqrt 13) / 2 ∧ 
    x2 = (-1 - Real.sqrt 13) / 2 :=
sorry

-- Theorem for the second equation
theorem second_eq_solutions :
  ∃ x1 x2 : ℝ, 
    second_eq x1 ∧ 
    second_eq x2 ∧ 
    x1 = -1/2 ∧ 
    x2 = 1 :=
sorry

end quadratic_solutions_second_eq_solutions_l185_18503


namespace range_f_a_2_range_a_two_zeros_l185_18527

-- Define the function f(x) = x^2 - ax - a + 3
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a + 3

-- Part 1: Range of f(x) when a = 2 and x ∈ [0, 3]
theorem range_f_a_2 :
  ∀ y ∈ Set.Icc 0 4, ∃ x ∈ Set.Icc 0 3, f 2 x = y :=
sorry

-- Part 2: Range of a when f(x) has two zeros x₁ and x₂ with x₁x₂ > 0
theorem range_a_two_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ * x₂ > 0) →
  a ∈ Set.Ioi (-6) ∪ Set.Ioo 2 3 :=
sorry

end range_f_a_2_range_a_two_zeros_l185_18527


namespace prob_spade_seven_red_l185_18595

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of sevens in a standard deck -/
def NumSevens : ℕ := 4

/-- Number of red cards in a standard deck -/
def NumRed : ℕ := 26

/-- Probability of drawing a spade, then a 7, then a red card from a standard 52-card deck -/
theorem prob_spade_seven_red (deck : ℕ) (spades : ℕ) (sevens : ℕ) (red : ℕ) :
  deck = StandardDeck →
  spades = NumSpades →
  sevens = NumSevens →
  red = NumRed →
  (spades / deck) * (sevens / (deck - 1)) * (red / (deck - 2)) = 1 / 100 := by
  sorry

end prob_spade_seven_red_l185_18595


namespace cyclists_speed_l185_18548

theorem cyclists_speed (initial_distance : ℝ) (fly_speed : ℝ) (fly_distance : ℝ) :
  initial_distance = 50 ∧ 
  fly_speed = 15 ∧ 
  fly_distance = 37.5 →
  ∃ (cyclist_speed : ℝ),
    cyclist_speed = 10 ∧ 
    initial_distance = 2 * cyclist_speed * (fly_distance / fly_speed) :=
by sorry

end cyclists_speed_l185_18548


namespace C_grazed_for_4_months_l185_18535

/-- The number of milkmen who rented the pasture -/
def num_milkmen : ℕ := 4

/-- The number of cows grazed by milkman A -/
def cows_A : ℕ := 24

/-- The number of months milkman A grazed his cows -/
def months_A : ℕ := 3

/-- The number of cows grazed by milkman B -/
def cows_B : ℕ := 10

/-- The number of months milkman B grazed his cows -/
def months_B : ℕ := 5

/-- The number of cows grazed by milkman C -/
def cows_C : ℕ := 35

/-- The number of cows grazed by milkman D -/
def cows_D : ℕ := 21

/-- The number of months milkman D grazed his cows -/
def months_D : ℕ := 3

/-- A's share of the rent in rupees -/
def share_A : ℕ := 720

/-- The total rent of the field in rupees -/
def total_rent : ℕ := 3250

/-- The theorem stating that C grazed his cows for 4 months -/
theorem C_grazed_for_4_months :
  ∃ (months_C : ℕ),
    months_C = 4 ∧
    total_rent = share_A +
      (cows_B * months_B * share_A / (cows_A * months_A)) +
      (cows_C * months_C * share_A / (cows_A * months_A)) +
      (cows_D * months_D * share_A / (cows_A * months_A)) :=
by sorry

end C_grazed_for_4_months_l185_18535


namespace parabola_max_distance_to_point_l185_18592

/-- The parabola C: y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line on the 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The distance between a point and a line -/
def distance_point_to_line (pt : Point) (l : Line) : ℝ :=
  sorry

theorem parabola_max_distance_to_point 
  (C : Parabola) 
  (A : Point)
  (hA : A.x = -1/2 ∧ A.y = 1/2)
  (hAxis : A.x = -C.p/2)
  (M N : Point)
  (hM : M.y^2 = 2 * C.p * M.x)
  (hN : N.y^2 = 2 * C.p * N.x)
  (hMN : M.y * N.y < 0)
  (O : Point)
  (hO : O.x = 0 ∧ O.y = 0)
  (hDot : (M.x - O.x) * (N.x - O.x) + (M.y - O.y) * (N.y - O.y) = 3)
  : ∃ (l : Line), ∀ (l' : Line), 
    (∃ (P Q : Point), P.y^2 = 2 * C.p * P.x ∧ Q.y^2 = 2 * C.p * Q.x ∧ P.y * Q.y < 0 ∧ 
      P.y = l'.slope * P.x + l'.intercept ∧ Q.y = l'.slope * Q.x + l'.intercept) →
    distance_point_to_line A l ≥ distance_point_to_line A l' ∧
    distance_point_to_line A l = 5 * Real.sqrt 2 / 2 :=
sorry

end parabola_max_distance_to_point_l185_18592


namespace sweater_vest_to_shirt_ratio_l185_18580

/-- Represents Carlton's wardrobe and outfit combinations -/
structure Wardrobe where
  sweater_vests : ℕ
  button_up_shirts : ℕ
  outfits : ℕ

/-- The ratio of sweater vests to button-up shirts is 2:1 given the conditions -/
theorem sweater_vest_to_shirt_ratio (w : Wardrobe) 
  (h1 : w.button_up_shirts = 3)
  (h2 : w.outfits = 18)
  (h3 : w.outfits = w.sweater_vests * w.button_up_shirts) :
  w.sweater_vests / w.button_up_shirts = 2 := by
  sorry

#check sweater_vest_to_shirt_ratio

end sweater_vest_to_shirt_ratio_l185_18580


namespace special_polynomial_form_l185_18582

/-- A polynomial satisfying the given conditions -/
class SpecialPolynomial (P : ℝ → ℝ) where
  zero_condition : P 0 = 0
  functional_equation : ∀ x : ℝ, P x = (1/2) * (P (x + 1) + P (x - 1))

/-- Theorem stating that any polynomial satisfying the given conditions is of the form P(x) = ax -/
theorem special_polynomial_form {P : ℝ → ℝ} [SpecialPolynomial P] : 
  ∃ a : ℝ, ∀ x : ℝ, P x = a * x := by
  sorry

end special_polynomial_form_l185_18582


namespace farthest_point_is_two_zero_l185_18575

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0

-- Define the tangency conditions
def externally_tangent (x y : ℝ) : Prop := 
  ∃ (r : ℝ), r > 0 ∧ ∀ (x' y' : ℝ), circle1 x' y' → ((x - x')^2 + (y - y')^2 = (1 + r)^2)

def internally_tangent (x y : ℝ) : Prop := 
  ∃ (r : ℝ), r > 0 ∧ ∀ (x' y' : ℝ), circle2 x' y' → ((x - x')^2 + (y - y')^2 = (3 - r)^2)

-- Define the farthest point condition
def is_farthest_point (x y : ℝ) : Prop :=
  externally_tangent x y ∧ internally_tangent x y ∧
  ∀ (x' y' : ℝ), externally_tangent x' y' → internally_tangent x' y' → 
    (x^2 + y^2 ≥ x'^2 + y'^2)

-- Theorem statement
theorem farthest_point_is_two_zero : is_farthest_point 2 0 := by sorry

end farthest_point_is_two_zero_l185_18575


namespace basketball_points_third_game_l185_18534

theorem basketball_points_third_game 
  (total_points : ℕ) 
  (first_game_fraction : ℚ) 
  (second_game_fraction : ℚ) 
  (h1 : total_points = 20) 
  (h2 : first_game_fraction = 1/2) 
  (h3 : second_game_fraction = 1/10) : 
  total_points - (first_game_fraction * total_points + second_game_fraction * total_points) = 8 := by
  sorry

end basketball_points_third_game_l185_18534


namespace solution_is_eight_l185_18547

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  lg (2^x + 2*x - 16) = x * (1 - lg 5)

-- Theorem statement
theorem solution_is_eight : 
  ∃ (x : ℝ), equation x ∧ x = 8 :=
sorry

end solution_is_eight_l185_18547


namespace absolute_value_ratio_l185_18581

theorem absolute_value_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + y^2 = 5*x*y) : 
  |((x+y)/(x-y))| = Real.sqrt (7/3) := by
sorry

end absolute_value_ratio_l185_18581


namespace expression_simplification_l185_18567

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2 - x) + 4*(2 + 3*x) - 5*(1 - 2*x) = 27*x - 3 := by
  sorry

end expression_simplification_l185_18567


namespace worst_player_is_daughter_l185_18563

-- Define the set of players
inductive Player
| Father
| Sister
| Daughter
| Son

-- Define the gender type
inductive Gender
| Male
| Female

-- Define the generation type
inductive Generation
| Older
| Younger

-- Function to get the gender of a player
def gender : Player → Gender
| Player.Father => Gender.Male
| Player.Sister => Gender.Female
| Player.Daughter => Gender.Female
| Player.Son => Gender.Male

-- Function to get the generation of a player
def generation : Player → Generation
| Player.Father => Generation.Older
| Player.Sister => Generation.Older
| Player.Daughter => Generation.Younger
| Player.Son => Generation.Younger

-- Function to determine if two players could be twins
def couldBeTwins : Player → Player → Prop
| Player.Daughter, Player.Son => True
| Player.Son, Player.Daughter => True
| _, _ => False

-- Theorem statement
theorem worst_player_is_daughter :
  ∀ (worst best : Player),
    (∃ twin : Player, couldBeTwins worst twin ∧ gender twin = gender best) →
    generation worst ≠ generation best →
    worst = Player.Daughter :=
sorry

end worst_player_is_daughter_l185_18563


namespace smallest_sum_with_conditions_l185_18524

def is_relatively_prime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem smallest_sum_with_conditions :
  ∃ (a b c d e : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    ¬(is_relatively_prime a b) ∧
    ¬(is_relatively_prime b c) ∧
    ¬(is_relatively_prime c d) ∧
    ¬(is_relatively_prime d e) ∧
    is_relatively_prime a c ∧
    is_relatively_prime a d ∧
    is_relatively_prime a e ∧
    is_relatively_prime b d ∧
    is_relatively_prime b e ∧
    is_relatively_prime c e ∧
    a + b + c + d + e = 75 ∧
    (∀ (a' b' c' d' e' : ℕ),
      a' > 0 → b' > 0 → c' > 0 → d' > 0 → e' > 0 →
      ¬(is_relatively_prime a' b') →
      ¬(is_relatively_prime b' c') →
      ¬(is_relatively_prime c' d') →
      ¬(is_relatively_prime d' e') →
      is_relatively_prime a' c' →
      is_relatively_prime a' d' →
      is_relatively_prime a' e' →
      is_relatively_prime b' d' →
      is_relatively_prime b' e' →
      is_relatively_prime c' e' →
      a' + b' + c' + d' + e' ≥ 75) :=
by sorry

end smallest_sum_with_conditions_l185_18524


namespace white_balls_count_l185_18561

theorem white_balls_count (red_balls : ℕ) (white_balls : ℕ) :
  red_balls = 4 →
  (white_balls : ℚ) / (red_balls + white_balls : ℚ) = 2 / 3 →
  white_balls = 8 := by
sorry

end white_balls_count_l185_18561


namespace kvass_price_after_increases_l185_18571

theorem kvass_price_after_increases (x y : ℝ) : 
  x + y = 1 →
  1.2 * (0.5 * x + y) = 1 →
  1.44 * y < 1 :=
by sorry

end kvass_price_after_increases_l185_18571


namespace unique_congruence_in_range_l185_18593

theorem unique_congruence_in_range : ∃! n : ℤ, 3 ≤ n ∧ n ≤ 7 ∧ n ≡ 12345 [ZMOD 4] := by
  sorry

end unique_congruence_in_range_l185_18593


namespace absent_days_calculation_l185_18598

/-- Represents the contract details and outcome -/
structure ContractDetails where
  totalDays : ℕ
  paymentPerDay : ℚ
  finePerDay : ℚ
  totalReceived : ℚ

/-- Calculates the number of absent days based on contract details -/
def absentDays (contract : ContractDetails) : ℚ :=
  (contract.totalDays * contract.paymentPerDay - contract.totalReceived) / (contract.paymentPerDay + contract.finePerDay)

/-- Theorem stating that given the specific contract details, the number of absent days is 8 -/
theorem absent_days_calculation (contract : ContractDetails) 
  (h1 : contract.totalDays = 30)
  (h2 : contract.paymentPerDay = 25)
  (h3 : contract.finePerDay = 7.5)
  (h4 : contract.totalReceived = 490) :
  absentDays contract = 8 := by
  sorry

#eval absentDays { totalDays := 30, paymentPerDay := 25, finePerDay := 7.5, totalReceived := 490 }

end absent_days_calculation_l185_18598


namespace discount_difference_l185_18590

theorem discount_difference : 
  let original_bill : ℝ := 12000
  let single_discount_rate : ℝ := 0.35
  let first_successive_discount_rate : ℝ := 0.30
  let second_successive_discount_rate : ℝ := 0.06
  let single_discount_amount : ℝ := original_bill * (1 - single_discount_rate)
  let successive_discount_amount : ℝ := original_bill * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)
  successive_discount_amount - single_discount_amount = 96 := by
sorry

end discount_difference_l185_18590


namespace line_passes_through_P_and_intersects_l_l185_18501

-- Define the point P
def P : ℝ × ℝ := (0, 2)

-- Define the line l
def l (x y : ℝ) : Prop := x + 2 * y - 1 = 0

-- Define the line we found
def found_line (x y : ℝ) : Prop := y = x + 2

-- Theorem statement
theorem line_passes_through_P_and_intersects_l :
  -- The line passes through P
  found_line P.1 P.2 ∧
  -- The line is not parallel to l (they intersect)
  ∃ x y, found_line x y ∧ l x y ∧ (x, y) ≠ P :=
sorry

end line_passes_through_P_and_intersects_l_l185_18501


namespace cubic_equation_integer_solutions_l185_18541

theorem cubic_equation_integer_solutions :
  ∀ x y : ℤ, y^3 = x^3 + 8*x^2 - 6*x + 8 ↔ (x = 0 ∧ y = 2) ∨ (x = 9 ∧ y = 11) :=
by sorry

end cubic_equation_integer_solutions_l185_18541


namespace quadratic_root_inequality_l185_18576

theorem quadratic_root_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) 
  (h3 : a * 1^2 + b * 1 + c = 0) : -2 ≤ c / a ∧ c / a ≤ -1/2 := by
  sorry

end quadratic_root_inequality_l185_18576


namespace circle_m_range_l185_18594

-- Define the equation as a function of x, y, and m
def circle_equation (x y m : ℝ) : ℝ := x^2 + y^2 - 2*m*x + 2*m^2 + 2*m - 3

-- Define what it means for the equation to represent a circle
def is_circle (m : ℝ) : Prop :=
  ∃ (a b r : ℝ), r > 0 ∧ ∀ (x y : ℝ),
    circle_equation x y m = 0 ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem stating the range of m for which the equation represents a circle
theorem circle_m_range :
  ∀ m : ℝ, is_circle m ↔ -3 < m ∧ m < 1/2 := by sorry

end circle_m_range_l185_18594


namespace powder_division_theorem_l185_18537

/-- Represents the measurements and properties of the magical powder division problem. -/
structure PowderDivision where
  total_measured : ℝ
  remaining_measured : ℝ
  removed_measured : ℝ
  error : ℝ

/-- The actual weights of the two portions of the magical powder. -/
def actual_weights (pd : PowderDivision) : ℝ × ℝ :=
  (pd.remaining_measured - pd.error, pd.removed_measured - pd.error)

/-- Theorem stating that given the measurements and assuming a consistent error,
    the actual weights of the two portions are 4 and 3 zolotniks. -/
theorem powder_division_theorem (pd : PowderDivision) 
  (h1 : pd.total_measured = 6)
  (h2 : pd.remaining_measured = 3)
  (h3 : pd.removed_measured = 2)
  (h4 : pd.total_measured = pd.remaining_measured + pd.removed_measured - pd.error) :
  actual_weights pd = (4, 3) := by
  sorry

#eval actual_weights { total_measured := 6, remaining_measured := 3, removed_measured := 2, error := -1 }

end powder_division_theorem_l185_18537


namespace smallest_power_of_ten_minus_one_divisible_by_37_l185_18519

theorem smallest_power_of_ten_minus_one_divisible_by_37 :
  (∃ n : ℕ, 10^n - 1 ≡ 0 [MOD 37]) ∧
  (∀ k : ℕ, k < 3 → ¬(10^k - 1 ≡ 0 [MOD 37])) ∧
  (10^3 - 1 ≡ 0 [MOD 37]) :=
sorry

end smallest_power_of_ten_minus_one_divisible_by_37_l185_18519


namespace ratio_equality_l185_18583

theorem ratio_equality (a b c : ℝ) (h : a/2 = b/3 ∧ b/3 = c/4) : (a + b) / c = 5/4 := by
  sorry

end ratio_equality_l185_18583


namespace div_power_eq_reciprocal_power_l185_18508

/-- Division power operation for rational numbers -/
def div_power (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1 else 1 / a^(n - 1)

/-- Theorem: Division power equals reciprocal of power with exponent decreased by 2 -/
theorem div_power_eq_reciprocal_power (a : ℚ) (n : ℕ) (h : a ≠ 0) (hn : n ≥ 2) :
  div_power a n = 1 / a^(n - 2) := by
  sorry

end div_power_eq_reciprocal_power_l185_18508


namespace min_perimeter_isosceles_triangles_l185_18530

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * (((t.leg : ℝ) ^ 2 - ((t.base : ℝ) / 2) ^ 2).sqrt) / 2

/-- Theorem stating the minimum perimeter of two noncongruent isosceles triangles
    with the same area and bases in the ratio 3:2 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    area t1 = area t2 ∧
    t1.base * 2 = t2.base * 3 ∧
    perimeter t1 = perimeter t2 ∧
    perimeter t1 = 508 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      area s1 = area s2 →
      s1.base * 2 = s2.base * 3 →
      perimeter s1 = perimeter s2 →
      perimeter s1 ≥ 508) :=
by sorry

end min_perimeter_isosceles_triangles_l185_18530


namespace red_balls_count_l185_18543

theorem red_balls_count (total_balls : ℕ) (black_balls : ℕ) (prob_black : ℚ) : 
  black_balls = 5 → 
  prob_black = 1/4 → 
  total_balls = black_balls + (total_balls - black_balls) →
  (total_balls - black_balls) = 15 := by
  sorry

end red_balls_count_l185_18543


namespace vector_expression_evaluation_l185_18572

/-- Prove that the vector expression evaluates to the given result -/
theorem vector_expression_evaluation :
  (⟨3, -8⟩ : ℝ × ℝ) - 5 • (⟨2, -4⟩ : ℝ × ℝ) = (⟨-7, 12⟩ : ℝ × ℝ) := by
  sorry

end vector_expression_evaluation_l185_18572


namespace candies_per_packet_is_18_l185_18549

/-- The number of candies in each packet -/
def candies_per_packet : ℕ := 18

/-- The number of packets Bobby has -/
def num_packets : ℕ := 2

/-- The number of days Bobby eats 2 candies -/
def days_eating_two : ℕ := 5

/-- The number of days Bobby eats 1 candy -/
def days_eating_one : ℕ := 2

/-- The number of weeks it takes to finish the packets -/
def weeks_to_finish : ℕ := 3

/-- Theorem stating that the number of candies in each packet is 18 -/
theorem candies_per_packet_is_18 :
  candies_per_packet * num_packets = 
    (days_eating_two * 2 + days_eating_one) * weeks_to_finish :=
by sorry

end candies_per_packet_is_18_l185_18549


namespace line_tangent_to_parabola_l185_18517

/-- A line is tangent to a parabola if and only if the resulting quadratic equation has a double root --/
axiom tangent_iff_double_root (a b c : ℝ) : 
  (∃ k, a * k^2 + b * k + c = 0 ∧ b^2 - 4*a*c = 0) ↔ 
  (∃! x y : ℝ, a * x^2 + b * x + c = 0 ∧ y^2 = 4 * a * x)

/-- The main theorem: if the line 4x + 7y + k = 0 is tangent to the parabola y^2 = 16x, then k = 49 --/
theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4*x + 7*y + k = 0 → y^2 = 16*x) →
  (∃! x y : ℝ, 4*x + 7*y + k = 0 ∧ y^2 = 16*x) →
  k = 49 := by
  sorry


end line_tangent_to_parabola_l185_18517


namespace not_always_externally_tangent_l185_18520

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the positional relationship between two circles
inductive CircleRelationship
  | Disjoint
  | ExternallyTangent
  | Intersecting
  | InternallyTangent
  | Contained

-- Define a function to determine if two circles have no intersection
def noIntersection (c1 c2 : Circle) : Prop :=
  sorry

-- Define a function to determine the relationship between two circles
def circleRelationship (c1 c2 : Circle) : CircleRelationship :=
  sorry

-- Theorem statement
theorem not_always_externally_tangent (c1 c2 : Circle) :
  ¬(noIntersection c1 c2 → circleRelationship c1 c2 = CircleRelationship.ExternallyTangent) :=
sorry

end not_always_externally_tangent_l185_18520


namespace set_relations_l185_18573

def A (k : ℝ) : Set ℝ := {x | k * x^2 - 2 * x + 6 * k < 0 ∧ k ≠ 0}

theorem set_relations (k : ℝ) :
  (A k ⊆ Set.Ioo 2 3 → k ≥ 2/5) ∧
  (Set.Ioo 2 3 ⊆ A k → k ≤ 2/5) ∧
  (Set.inter (A k) (Set.Ioo 2 3) ≠ ∅ → k < Real.sqrt 6 / 6) :=
sorry

end set_relations_l185_18573


namespace jeans_price_proof_l185_18560

/-- The original price of one pair of jeans -/
def original_price : ℝ := 40

/-- The discounted price for two pairs of jeans -/
def discounted_price (p : ℝ) : ℝ := 2 * p * 0.9

/-- The total price for three pairs of jeans -/
def total_price (p : ℝ) : ℝ := discounted_price p + p

theorem jeans_price_proof :
  total_price original_price = 112 :=
sorry

end jeans_price_proof_l185_18560


namespace max_fraction_over65_l185_18504

/-- Represents the number of people in a room with age-related conditions -/
structure RoomPopulation where
  total : ℕ
  under21 : ℕ
  over65 : ℕ
  h1 : under21 = (3 * total) / 7
  h2 : 50 < total
  h3 : total < 100
  h4 : under21 = 30

/-- The maximum fraction of people over 65 in the room is 4/7 -/
theorem max_fraction_over65 (room : RoomPopulation) :
  (room.over65 : ℚ) / room.total ≤ 4 / 7 := by
  sorry

end max_fraction_over65_l185_18504


namespace nested_fraction_equals_21_55_l185_18507

theorem nested_fraction_equals_21_55 :
  1 / (3 - 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 21 / 55 := by
  sorry

end nested_fraction_equals_21_55_l185_18507


namespace two_points_imply_line_in_plane_l185_18532

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define what it means for a point to be on a line
variable (on_line : Point → Line → Prop)

-- Define what it means for a point to be within a plane
variable (in_plane : Point → Plane → Prop)

-- Define what it means for a line to be within a plane
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem two_points_imply_line_in_plane 
  (a : Line) (α : Plane) (A B : Point) 
  (h1 : A ≠ B) 
  (h2 : on_line A a) 
  (h3 : on_line B a) 
  (h4 : in_plane A α) 
  (h5 : in_plane B α) : 
  line_in_plane a α :=
sorry

end two_points_imply_line_in_plane_l185_18532


namespace cory_chairs_proof_l185_18533

/-- The number of chairs Cory bought -/
def num_chairs : ℕ := 4

/-- The cost of the patio table -/
def table_cost : ℕ := 55

/-- The cost of each chair -/
def chair_cost : ℕ := 20

/-- The total cost of the table and chairs -/
def total_cost : ℕ := 135

theorem cory_chairs_proof :
  num_chairs * chair_cost + table_cost = total_cost :=
by sorry

end cory_chairs_proof_l185_18533


namespace increasing_odd_function_bound_l185_18529

/-- A function f: ℝ → ℝ is a "k-type increasing function" if for all x, f(x + k) > f(x) -/
def is_k_type_increasing (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x, f (x + k) > f x

theorem increasing_odd_function_bound (f : ℝ → ℝ) (a : ℝ) 
    (h_odd : ∀ x, f (-x) = -f x)
    (h_pos : ∀ x > 0, f x = |x - a| - 2*a)
    (h_inc : is_k_type_increasing f 2017) :
    a < 2017/6 := by
  sorry

end increasing_odd_function_bound_l185_18529


namespace total_grapes_is_83_l185_18506

/-- The number of grapes in Rob's bowl -/
def robs_grapes : ℕ := 25

/-- The number of grapes in Allie's bowl -/
def allies_grapes : ℕ := robs_grapes + 2

/-- The number of grapes in Allyn's bowl -/
def allyns_grapes : ℕ := allies_grapes + 4

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := robs_grapes + allies_grapes + allyns_grapes

theorem total_grapes_is_83 : total_grapes = 83 := by
  sorry

end total_grapes_is_83_l185_18506


namespace divisors_of_16n4_l185_18536

theorem divisors_of_16n4 (n : ℕ) (h_odd : Odd n) (h_divisors : (Nat.divisors n).card = 13) :
  (Nat.divisors (16 * n^4)).card = 245 :=
sorry

end divisors_of_16n4_l185_18536


namespace limit_of_S_is_infinity_l185_18587

def S (n : ℕ) : ℕ := (n + 1) * n / 2

theorem limit_of_S_is_infinity :
  ∀ M : ℝ, ∃ N : ℕ, ∀ n : ℕ, n ≥ N → (S n : ℝ) > M :=
sorry

end limit_of_S_is_infinity_l185_18587


namespace x_minus_y_values_l185_18552

theorem x_minus_y_values (x y : ℝ) (h1 : |x + 1| = 4) (h2 : (y + 2)^2 = 0) :
  x - y = 5 ∨ x - y = -3 := by
sorry

end x_minus_y_values_l185_18552


namespace sqrt_neg_nine_squared_l185_18518

theorem sqrt_neg_nine_squared : Real.sqrt ((-9)^2) = 9 := by sorry

end sqrt_neg_nine_squared_l185_18518


namespace find_t_l185_18585

theorem find_t (s t : ℚ) (eq1 : 8 * s + 7 * t = 95) (eq2 : s = 2 * t - 3) : t = 119 / 23 := by
  sorry

end find_t_l185_18585


namespace square_side_length_difference_l185_18538

/-- Given two squares with side lengths x and y, where the perimeter of the smaller square
    is 20 cm less than the perimeter of the larger square, prove that the side length of
    the larger square is 5 cm more than the side length of the smaller square. -/
theorem square_side_length_difference (x y : ℝ) (h : 4 * x + 20 = 4 * y) : y = x + 5 := by
  sorry

end square_side_length_difference_l185_18538


namespace smallest_number_divisible_l185_18544

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m ≥ 24 ∧ (m - 24) % 5 = 0 ∧ (m - 24) % 10 = 0 ∧ (m - 24) % 15 = 0 ∧ (m - 24) % 20 = 0 → m ≥ n) ∧
  n ≥ 24 ∧ (n - 24) % 5 = 0 ∧ (n - 24) % 10 = 0 ∧ (n - 24) % 15 = 0 ∧ (n - 24) % 20 = 0 →
  n = 84 :=
by sorry

end smallest_number_divisible_l185_18544


namespace bottles_needed_to_fill_container_l185_18597

def craft_bottle_volume : ℕ := 150
def decorative_container_volume : ℕ := 2650

theorem bottles_needed_to_fill_container : 
  ∃ n : ℕ, n * craft_bottle_volume ≥ decorative_container_volume ∧ 
  ∀ m : ℕ, m * craft_bottle_volume ≥ decorative_container_volume → n ≤ m :=
by sorry

end bottles_needed_to_fill_container_l185_18597


namespace min_non_acute_angles_l185_18586

/-- A convex polygon with 1992 sides -/
structure ConvexPolygon1992 where
  sides : ℕ
  convex : Bool
  sides_eq : sides = 1992
  is_convex : convex = true

/-- The number of interior angles that are not acute in a polygon -/
def non_acute_angles (p : ConvexPolygon1992) : ℕ := sorry

/-- The theorem stating the minimum number of non-acute angles in a ConvexPolygon1992 -/
theorem min_non_acute_angles (p : ConvexPolygon1992) : 
  non_acute_angles p ≥ 1989 := by sorry

end min_non_acute_angles_l185_18586


namespace train_crossing_time_l185_18568

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 180 →
  train_speed_kmh = 72 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end train_crossing_time_l185_18568


namespace smallest_multiple_one_to_five_l185_18542

theorem smallest_multiple_one_to_five : ∃ n : ℕ+, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 → i ∣ n) ∧ (∀ m : ℕ+, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 → i ∣ m) → n ≤ m) ∧ n = 60 := by
  sorry

end smallest_multiple_one_to_five_l185_18542


namespace no_rearrangement_sum_999999999_l185_18500

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Define a predicate to check if a number is a digit rearrangement of another
def isDigitRearrangement (n m : ℕ) : Prop :=
  sumOfDigits n = sumOfDigits m

theorem no_rearrangement_sum_999999999 (n : ℕ) :
  ¬∃ m : ℕ, isDigitRearrangement n m ∧ m + n = 999999999 :=
sorry

end no_rearrangement_sum_999999999_l185_18500


namespace bookkeeper_probability_l185_18562

def word_length : ℕ := 10

def num_e : ℕ := 3
def num_o : ℕ := 2
def num_k : ℕ := 2
def num_b : ℕ := 1
def num_p : ℕ := 1
def num_r : ℕ := 1

def adjacent_o : Prop := true
def two_adjacent_e : Prop := true
def no_o_e_at_beginning : Prop := true

def total_arrangements : ℕ := 9600

theorem bookkeeper_probability : 
  word_length = num_e + num_o + num_k + num_b + num_p + num_r →
  adjacent_o →
  two_adjacent_e →
  no_o_e_at_beginning →
  (1 : ℚ) / total_arrangements = (1 : ℚ) / 9600 :=
sorry

end bookkeeper_probability_l185_18562


namespace square_sum_problem_l185_18513

theorem square_sum_problem (a b c d : ℤ) (h : (a^2 + b^2) * (c^2 + d^2) = 1993) : a^2 + b^2 + c^2 + d^2 = 1994 := by
  sorry

-- Define 1993 as a prime number
def p : ℕ := 1993

axiom p_prime : Nat.Prime p

end square_sum_problem_l185_18513


namespace colored_segment_existence_l185_18502

/-- A color type with exactly 4 colors -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- A point on a line with a color -/
structure ColoredPoint where
  position : ℝ
  color : Color

/-- The theorem statement -/
theorem colored_segment_existence 
  (n : ℕ) 
  (points : Fin n → ColoredPoint) 
  (h_n : n ≥ 4) 
  (h_distinct : ∀ i j, i ≠ j → (points i).position ≠ (points j).position) 
  (h_all_colors : ∀ c : Color, ∃ i, (points i).color = c) :
  ∃ (i j : Fin n), i < j ∧
    (∃ (c₁ c₂ c₃ c₄ : Color), 
      c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₁ ≠ c₄ ∧ c₂ ≠ c₃ ∧ c₂ ≠ c₄ ∧ c₃ ≠ c₄ ∧
      (∃! k, i ≤ k ∧ k ≤ j ∧ (points k).color = c₁) ∧
      (∃! k, i ≤ k ∧ k ≤ j ∧ (points k).color = c₂) ∧
      (∃ k, i < k ∧ k < j ∧ (points k).color = c₃) ∧
      (∃ k, i < k ∧ k < j ∧ (points k).color = c₄)) :=
by
  sorry

end colored_segment_existence_l185_18502


namespace max_value_of_sum_l185_18574

theorem max_value_of_sum (a c d : ℤ) (b : ℕ+) 
  (h1 : a + b = c) 
  (h2 : b + c = d) 
  (h3 : c + d = a) : 
  (a + b + c + d : ℤ) ≤ -5 ∧ ∃ (a₀ c₀ d₀ : ℤ) (b₀ : ℕ+), 
    a₀ + b₀ = c₀ ∧ b₀ + c₀ = d₀ ∧ c₀ + d₀ = a₀ ∧ a₀ + b₀ + c₀ + d₀ = -5 :=
by
  sorry

end max_value_of_sum_l185_18574


namespace complex_expression_equals_four_l185_18577

theorem complex_expression_equals_four :
  (1/2)⁻¹ - Real.sqrt 3 * Real.tan (30 * π / 180) + (π - 2023)^0 + |-2| = 4 := by
  sorry

end complex_expression_equals_four_l185_18577


namespace certain_number_equation_l185_18526

theorem certain_number_equation : ∃ x : ℝ, 0.6 * 50 = 0.45 * x + 16.5 := by
  sorry

end certain_number_equation_l185_18526


namespace f_properties_l185_18531

def f (x : ℝ) := |2*x + 3| + |2*x - 1|

theorem f_properties :
  (∀ x : ℝ, f x < 10 ↔ x ∈ Set.Ioo (-3) 2) ∧
  (∀ a : ℝ, (∀ x : ℝ, f x ≥ |a - 1|) ↔ a ∈ Set.Icc (-2) 5) := by
  sorry

end f_properties_l185_18531


namespace frenchHorn_trombone_difference_l185_18589

/-- The number of band members for each instrument and their relationships --/
structure BandComposition where
  flute : ℕ
  trumpet : ℕ
  trombone : ℕ
  drums : ℕ
  clarinet : ℕ
  frenchHorn : ℕ
  fluteCount : flute = 5
  trumpetCount : trumpet = 3 * flute
  tromboneCount : trombone = trumpet - 8
  drumsCount : drums = trombone + 11
  clarinetCount : clarinet = 2 * flute
  frenchHornMoreThanTrombone : frenchHorn > trombone
  totalSeats : flute + trumpet + trombone + drums + clarinet + frenchHorn = 65

/-- The theorem stating the difference between French horn and trombone players --/
theorem frenchHorn_trombone_difference (b : BandComposition) :
  b.frenchHorn - b.trombone = 3 := by
  sorry

end frenchHorn_trombone_difference_l185_18589


namespace optimal_production_consumption_theorem_l185_18596

/-- Represents a country's production capabilities and consumption --/
structure Country where
  eggplant_production : ℝ
  corn_production : ℝ
  consumption : ℝ × ℝ

/-- The global market for agricultural products --/
structure Market where
  price : ℝ

/-- Calculates the optimal production and consumption for two countries --/
def optimal_production_and_consumption (a b : Country) (m : Market) : (Country × Country) :=
  sorry

/-- Main theorem: Optimal production and consumption for countries A and B --/
theorem optimal_production_consumption_theorem (a b : Country) (m : Market) :
  a.eggplant_production = 10 ∧
  a.corn_production = 8 ∧
  b.eggplant_production = 18 ∧
  b.corn_production = 12 ∧
  m.price > 0 →
  let (a', b') := optimal_production_and_consumption a b m
  a'.consumption = (4, 4) ∧ b'.consumption = (9, 9) :=
sorry

end optimal_production_consumption_theorem_l185_18596


namespace hex_fraction_sum_max_l185_18558

theorem hex_fraction_sum_max (a b c : ℕ) (y : ℕ) (h1 : a ≤ 15) (h2 : b ≤ 15) (h3 : c ≤ 15)
  (h4 : (a * 256 + b * 16 + c : ℕ) = 4096 / y) (h5 : 0 < y) (h6 : y ≤ 16) :
  a + b + c ≤ 1 :=
sorry

end hex_fraction_sum_max_l185_18558


namespace tile_arrangement_count_l185_18556

def brown_tiles : ℕ := 1
def purple_tiles : ℕ := 2
def green_tiles : ℕ := 3
def yellow_tiles : ℕ := 3

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

def distinguishable_arrangements : ℕ := total_tiles.factorial / (brown_tiles.factorial * purple_tiles.factorial * green_tiles.factorial * yellow_tiles.factorial)

theorem tile_arrangement_count : distinguishable_arrangements = 5040 := by
  sorry

end tile_arrangement_count_l185_18556


namespace greatest_common_multiple_under_120_l185_18545

theorem greatest_common_multiple_under_120 : 
  ∃ (n : ℕ), n = 90 ∧ 
  (∀ m : ℕ, m < 120 ∧ 9 ∣ m ∧ 15 ∣ m → m ≤ n) ∧
  9 ∣ n ∧ 15 ∣ n ∧ n < 120 :=
by sorry

end greatest_common_multiple_under_120_l185_18545


namespace function_value_proof_l185_18569

theorem function_value_proof (x : ℝ) (f : ℝ → ℝ) 
  (h1 : x > 0) 
  (h2 : x + 17 = 60 * f x) 
  (h3 : x = 3) : 
  f 3 = 1/3 := by
  sorry

end function_value_proof_l185_18569


namespace smallest_solution_of_equation_l185_18599

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 1 / (x - 3) + 1 / (x - 5) - 4 / (x - 4)
  ∃ (s : ℝ), s = 4 - Real.sqrt 2 ∧ 
    (f s = 0) ∧ 
    (∀ (x : ℝ), f x = 0 → x ≥ s) := by
  sorry

end smallest_solution_of_equation_l185_18599


namespace total_balls_l185_18510

def soccer_balls : ℕ := 20
def basketballs : ℕ := soccer_balls + 5
def tennis_balls : ℕ := 2 * soccer_balls
def baseballs : ℕ := soccer_balls + 10
def volleyballs : ℕ := 30

theorem total_balls : 
  soccer_balls + basketballs + tennis_balls + baseballs + volleyballs = 145 := by
  sorry

end total_balls_l185_18510


namespace solution_characterization_l185_18539

theorem solution_characterization (x y : ℝ) :
  (|x| + |y| = 1340) ∧ (x^3 + y^3 + 2010*x*y = 670^3) →
  (x + y = 670) ∧ (x * y = -673350) :=
by sorry

end solution_characterization_l185_18539


namespace completing_square_equivalence_l185_18546

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
by sorry

end completing_square_equivalence_l185_18546


namespace max_children_correct_l185_18551

/-- Represents the types of buses available --/
inductive BusType
| A
| B
| C
| D

/-- Calculates the total number of seats for a given bus type --/
def totalSeats (t : BusType) : ℕ :=
  match t with
  | BusType.A => 36
  | BusType.B => 54
  | BusType.C => 36
  | BusType.D => 36

/-- Represents the safety regulation for maximum number of children per bus type --/
def safetyRegulation (t : BusType) : ℕ :=
  match t with
  | BusType.A => 40
  | BusType.B => 50
  | BusType.C => 35
  | BusType.D => 30

/-- Calculates the maximum number of children that can be accommodated on a given bus type --/
def maxChildren (t : BusType) : ℕ :=
  min (totalSeats t) (safetyRegulation t)

theorem max_children_correct :
  (maxChildren BusType.A = 36) ∧
  (maxChildren BusType.B = 50) ∧
  (maxChildren BusType.C = 35) ∧
  (maxChildren BusType.D = 30) :=
by
  sorry


end max_children_correct_l185_18551


namespace seating_probability_l185_18512

/-- The number of chairs in the row -/
def total_chairs : ℕ := 10

/-- The number of usable chairs -/
def usable_chairs : ℕ := total_chairs - 1

/-- The probability that Mary and James do not sit next to each other -/
def probability_not_adjacent : ℚ := 7/9

theorem seating_probability :
  (total_chairs : ℕ) = 10 →
  (usable_chairs : ℕ) = total_chairs - 1 →
  probability_not_adjacent = 7/9 := by
  sorry

end seating_probability_l185_18512


namespace jakes_drink_volume_l185_18550

/-- Represents the composition of a drink in parts -/
structure DrinkComposition :=
  (coke : ℕ)
  (sprite : ℕ)
  (mountainDew : ℕ)

/-- Calculates the total volume of a drink given its composition and the volume of Coke -/
def totalVolume (composition : DrinkComposition) (cokeVolume : ℚ) : ℚ :=
  let totalParts := composition.coke + composition.sprite + composition.mountainDew
  let volumePerPart := cokeVolume / composition.coke
  totalParts * volumePerPart

/-- Theorem: The total volume of Jake's drink is 18 ounces -/
theorem jakes_drink_volume :
  let composition : DrinkComposition := ⟨2, 1, 3⟩
  let cokeVolume : ℚ := 6
  totalVolume composition cokeVolume = 18 := by
  sorry

end jakes_drink_volume_l185_18550


namespace bees_on_first_day_l185_18528

/-- Given that Mrs. Hilt saw some bees on the first day and 3 times as many on the second day,
    counting 432 bees on the second day, prove that she saw 144 bees on the first day. -/
theorem bees_on_first_day (first_day : ℕ) (second_day : ℕ) : 
  second_day = 3 * first_day → second_day = 432 → first_day = 144 := by
  sorry

end bees_on_first_day_l185_18528


namespace ceiling_negative_sqrt_64_over_9_l185_18559

theorem ceiling_negative_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by
  sorry

end ceiling_negative_sqrt_64_over_9_l185_18559


namespace extended_pattern_ratio_l185_18511

/-- Represents a square pattern of tiles -/
structure TilePattern :=
  (size : ℕ)
  (black_tiles : ℕ)
  (white_tiles : ℕ)

/-- Adds a border of white tiles to a given pattern -/
def add_border (pattern : TilePattern) : TilePattern :=
  { size := pattern.size + 2,
    black_tiles := pattern.black_tiles,
    white_tiles := pattern.white_tiles + (pattern.size + 2)^2 - pattern.size^2 }

/-- The ratio of black tiles to white tiles -/
def tile_ratio (pattern : TilePattern) : ℚ :=
  pattern.black_tiles / (pattern.black_tiles + pattern.white_tiles)

theorem extended_pattern_ratio :
  let initial_pattern : TilePattern := ⟨6, 12, 24⟩
  let extended_pattern := add_border initial_pattern
  tile_ratio extended_pattern = 3/13 := by
  sorry

end extended_pattern_ratio_l185_18511


namespace father_speed_is_60kmh_l185_18509

/-- Misha's father's driving speed in km/h -/
def father_speed : ℝ := 60

/-- Distance Misha walked in km -/
def misha_walk_distance : ℝ := 5

/-- Time saved in minutes -/
def time_saved : ℝ := 10

/-- Proves that Misha's father's driving speed is 60 km/h given the conditions -/
theorem father_speed_is_60kmh :
  father_speed = 60 ∧
  misha_walk_distance = 5 ∧
  time_saved = 10 →
  father_speed = 60 := by
  sorry

#check father_speed_is_60kmh

end father_speed_is_60kmh_l185_18509


namespace imaginary_part_of_complex_product_l185_18516

theorem imaginary_part_of_complex_product : Complex.im ((1 - Complex.I) * (3 + Complex.I)) = -2 := by
  sorry

end imaginary_part_of_complex_product_l185_18516


namespace max_surface_area_increase_l185_18515

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cuboid -/
def surfaceArea (d : CuboidDimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- The dimensions of the original small cuboid -/
def originalCuboid : CuboidDimensions :=
  { length := 3, width := 2, height := 1 }

/-- Theorem stating the maximum increase in surface area -/
theorem max_surface_area_increase :
  ∃ (finalCuboid : CuboidDimensions),
    surfaceArea finalCuboid - surfaceArea originalCuboid ≤ 10 ∧
    ∀ (otherCuboid : CuboidDimensions),
      surfaceArea otherCuboid - surfaceArea originalCuboid ≤
        surfaceArea finalCuboid - surfaceArea originalCuboid :=
by sorry

end max_surface_area_increase_l185_18515


namespace original_average_l185_18554

theorem original_average (n : ℕ) (a : ℝ) (h1 : n = 15) (h2 : (n * (a + 12)) / n = 52) : a = 40 := by
  sorry

end original_average_l185_18554


namespace peanut_box_count_l185_18584

/-- Given an initial quantity of peanuts in a box and an additional quantity added,
    compute the final quantity of peanuts in the box. -/
def final_peanut_count (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem stating that given 4 initial peanuts and 6 added peanuts, 
    the final count is 10 peanuts. -/
theorem peanut_box_count : final_peanut_count 4 6 = 10 := by
  sorry

end peanut_box_count_l185_18584


namespace share_distribution_l185_18540

theorem share_distribution (total : ℚ) (a b c : ℚ) : 
  total = 595 →
  a = (2/3) * b →
  b = (1/4) * c →
  a + b + c = total →
  a = 70 := by sorry

end share_distribution_l185_18540


namespace jean_grandchildren_gift_l185_18553

/-- Calculates the total amount given to grandchildren per year -/
def total_given_to_grandchildren (num_grandchildren : ℕ) (cards_per_grandchild : ℕ) (amount_per_card : ℕ) : ℕ :=
  num_grandchildren * cards_per_grandchild * amount_per_card

/-- Proves that Jean gives $480 to her grandchildren per year -/
theorem jean_grandchildren_gift :
  total_given_to_grandchildren 3 2 80 = 480 := by
  sorry

#eval total_given_to_grandchildren 3 2 80

end jean_grandchildren_gift_l185_18553


namespace shoe_pairs_l185_18522

theorem shoe_pairs (ellie riley : ℕ) : 
  ellie = riley + 3 →
  ellie + riley = 13 →
  ellie = 8 := by
sorry

end shoe_pairs_l185_18522


namespace fraction_sum_simplification_l185_18591

theorem fraction_sum_simplification :
  2 / 520 + 23 / 40 = 301 / 520 := by
sorry

end fraction_sum_simplification_l185_18591


namespace expression_value_l185_18525

theorem expression_value (x y : ℚ) (h : 12 * x = 4 * y + 2) :
  6 * y - 18 * x + 7 = 4 := by sorry

end expression_value_l185_18525


namespace hyperbola_equation_from_asymptotes_and_point_l185_18521

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ

/-- The equation of a hyperbola given its asymptotes and a point it passes through -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => x^2 / 3 - y^2 / 12 = 1

/-- Theorem: Given a hyperbola with asymptotes y = ±2x and passing through (2, 2),
    its equation is x²/3 - y²/12 = 1 -/
theorem hyperbola_equation_from_asymptotes_and_point :
  ∀ (h : Hyperbola), h.asymptote_slope = 2 → h.point = (2, 2) →
  ∀ x y, hyperbola_equation h x y ↔ x^2 / 3 - y^2 / 12 = 1 :=
sorry

end hyperbola_equation_from_asymptotes_and_point_l185_18521


namespace truck_loading_time_l185_18523

theorem truck_loading_time (rate1 rate2 : ℚ) (h1 : rate1 = 1 / 6) (h2 : rate2 = 1 / 5) :
  1 / (rate1 + rate2) = 30 / 11 := by
  sorry

end truck_loading_time_l185_18523


namespace no_real_solutions_l185_18565

theorem no_real_solutions :
  ∀ x : ℝ, (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 20 * x^9 :=
by sorry

end no_real_solutions_l185_18565
