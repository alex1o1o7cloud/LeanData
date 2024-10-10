import Mathlib

namespace four_white_possible_l4067_406759

/-- Represents the state of the urn -/
structure UrnState :=
  (white : ℕ)
  (black : ℕ)

/-- Represents the possible operations on the urn -/
inductive Operation
  | removeFourBlackAddTwoBlack
  | removeThreeBlackOneWhiteAddOneBlackOneWhite
  | removeOneBlackThreeWhiteAddTwoWhite
  | removeFourWhiteAddTwoWhiteOneBlack

/-- Applies an operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.removeFourBlackAddTwoBlack => 
      ⟨state.white, state.black - 2⟩
  | Operation.removeThreeBlackOneWhiteAddOneBlackOneWhite => 
      ⟨state.white, state.black - 2⟩
  | Operation.removeOneBlackThreeWhiteAddTwoWhite => 
      ⟨state.white - 1, state.black - 1⟩
  | Operation.removeFourWhiteAddTwoWhiteOneBlack => 
      ⟨state.white - 2, state.black + 1⟩

/-- The theorem to be proved -/
theorem four_white_possible : 
  ∃ (ops : List Operation), 
    let final_state := ops.foldl applyOperation ⟨150, 150⟩
    final_state.white = 4 :=
sorry

end four_white_possible_l4067_406759


namespace logarithm_simplification_l4067_406753

theorem logarithm_simplification :
  (Real.log 2 / Real.log 6)^2 + (Real.log 2 / Real.log 6) * (Real.log 3 / Real.log 6) +
  2 * (Real.log 3 / Real.log 6) - 6^(Real.log 2 / Real.log 6) = -(Real.log 2 / Real.log 6) := by
  sorry

end logarithm_simplification_l4067_406753


namespace complex_number_sum_l4067_406746

theorem complex_number_sum (a b : ℝ) : 
  (Complex.I : ℂ)^5 * (Complex.I - 1) = Complex.mk a b → a + b = -2 := by
  sorry

end complex_number_sum_l4067_406746


namespace b_investment_is_8000_l4067_406770

/-- Represents a partnership with three partners -/
structure Partnership where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  a_profit : ℝ
  b_profit : ℝ

/-- The profit share is proportional to the investment -/
def profit_proportional (p : Partnership) : Prop :=
  p.a_profit / p.a_investment = p.b_profit / p.b_investment

/-- Theorem stating that given the conditions, b's investment is $8000 -/
theorem b_investment_is_8000 (p : Partnership) 
  (h1 : p.a_investment = 7000)
  (h2 : p.c_investment = 18000)
  (h3 : p.a_profit = 560)
  (h4 : p.b_profit = 880)
  (h5 : profit_proportional p) : 
  p.b_investment = 8000 := by
  sorry

end b_investment_is_8000_l4067_406770


namespace replacement_cost_20_gyms_l4067_406784

/-- The cost to replace all cardio machines in multiple gyms -/
def total_replacement_cost (num_gyms : ℕ) (bike_cost : ℕ) : ℕ :=
  let treadmill_cost : ℕ := (3 * bike_cost) / 2
  let elliptical_cost : ℕ := 2 * treadmill_cost
  let gym_cost : ℕ := 10 * bike_cost + 5 * treadmill_cost + 5 * elliptical_cost
  num_gyms * gym_cost

/-- Theorem stating the total replacement cost for 20 gyms -/
theorem replacement_cost_20_gyms :
  total_replacement_cost 20 700 = 455000 := by
  sorry

end replacement_cost_20_gyms_l4067_406784


namespace curve_self_intersection_l4067_406764

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := t^2 - 3

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^3 - 6*t + 4

/-- The curve intersects itself if there exist two distinct real numbers that yield the same point -/
def self_intersection (a b : ℝ) : Prop :=
  a ≠ b ∧ x a = x b ∧ y a = y b

theorem curve_self_intersection :
  ∃ a b : ℝ, self_intersection a b ∧ x a = 3 ∧ y a = 4 :=
sorry

end curve_self_intersection_l4067_406764


namespace intersection_range_l4067_406755

-- Define the points P and Q
def P : ℝ × ℝ := (-1, 1)
def Q : ℝ × ℝ := (2, 2)

-- Define the line equation
def line_equation (m : ℝ) (x y : ℝ) : Prop := x + m * y + m = 0

-- Define the line PQ
def line_PQ (x y : ℝ) : Prop := y = x + 2

-- Theorem statement
theorem intersection_range :
  ∀ m : ℝ, 
  (∃ x y : ℝ, x > 2 ∧ line_equation m x y ∧ line_PQ x y) ↔ 
  -3 < m ∧ m < 0 :=
sorry

end intersection_range_l4067_406755


namespace largest_square_multiple_18_under_500_l4067_406747

theorem largest_square_multiple_18_under_500 : ∃ n : ℕ, 
  n^2 = 324 ∧ 
  18 ∣ n^2 ∧ 
  n^2 < 500 ∧ 
  ∀ m : ℕ, (m^2 > n^2 ∧ 18 ∣ m^2) → m^2 ≥ 500 :=
by sorry

end largest_square_multiple_18_under_500_l4067_406747


namespace parabola_equation_l4067_406700

/-- A parabola with focus on the x-axis passing through the point (1, 2) -/
structure Parabola where
  /-- The equation of the parabola in the form y^2 = 2px -/
  equation : ℝ → ℝ → Prop
  /-- The parabola passes through the point (1, 2) -/
  passes_through_point : equation 1 2
  /-- The focus of the parabola is on the x-axis -/
  focus_on_x_axis : ∃ p : ℝ, ∀ x y : ℝ, equation x y ↔ y^2 = 2*p*x

/-- The standard equation of the parabola is y^2 = 4x -/
theorem parabola_equation (p : Parabola) : 
  ∃ (f : ℝ → ℝ → Prop), (∀ x y : ℝ, f x y ↔ y^2 = 4*x) ∧ p.equation = f := by
  sorry

end parabola_equation_l4067_406700


namespace expectation_of_specific_distribution_l4067_406725

/-- The expected value of a random variable with a specific probability distribution -/
theorem expectation_of_specific_distribution (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h_sum : x + y + x = 1) :
  let ξ : ℝ → ℝ := fun ω => 
    if ω < x then 1
    else if ω < x + y then 2
    else 3
  2 = ∫ ω in Set.Icc 0 1, ξ ω ∂volume :=
by sorry

end expectation_of_specific_distribution_l4067_406725


namespace shared_course_count_is_24_l4067_406742

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of courses available -/
def total_courses : ℕ := 4

/-- The number of courses each person chooses -/
def courses_per_person : ℕ := 2

/-- The number of ways in which exactly one course is chosen by both people -/
def shared_course_count : ℕ := 
  choose total_courses courses_per_person * choose total_courses courses_per_person -
  choose total_courses courses_per_person -
  choose total_courses courses_per_person

theorem shared_course_count_is_24 : shared_course_count = 24 := by sorry

end shared_course_count_is_24_l4067_406742


namespace min_ab_for_line_through_point_l4067_406738

/-- Given a line equation (x/a) + (y/b) = 1 where a > 0 and b > 0,
    and the line passes through the point (1,1),
    the minimum value of ab is 4. -/
theorem min_ab_for_line_through_point (a b : ℝ) : 
  a > 0 → b > 0 → (1 / a + 1 / b = 1) → (∀ x y : ℝ, x / a + y / b = 1 → (x, y) = (1, 1)) → 
  ∀ c d : ℝ, c > 0 → d > 0 → (1 / c + 1 / d = 1) → c * d ≥ 4 := by
  sorry

#check min_ab_for_line_through_point

end min_ab_for_line_through_point_l4067_406738


namespace sqrt_difference_l4067_406771

theorem sqrt_difference (a b : ℝ) (ha : a < 0) (hb : b < 0) (hab : a - b = 6) :
  Real.sqrt (a^2) - Real.sqrt (b^2) = -6 := by
  sorry

end sqrt_difference_l4067_406771


namespace third_grade_sample_size_l4067_406780

/-- Calculates the number of students to be sampled from a specific grade in stratified sampling -/
def stratified_sample_size (total_students : ℕ) (sample_size : ℕ) (grade_students : ℕ) : ℕ :=
  (grade_students * sample_size) / total_students

/-- Theorem: In a stratified sampling of 65 students from a high school with 1300 total students,
    where 500 students are in the third grade, the number of students to be sampled from the
    third grade is 25. -/
theorem third_grade_sample_size :
  stratified_sample_size 1300 65 500 = 25 := by
  sorry

#eval stratified_sample_size 1300 65 500

end third_grade_sample_size_l4067_406780


namespace word_count_between_czyeb_and_xceda_l4067_406756

/-- Represents the set of available letters --/
inductive Letter : Type
  | A | B | C | D | E | X | Y | Z

/-- A word is a list of 5 letters --/
def Word := List Letter

/-- Convert a letter to its corresponding digit in base 8 --/
def letterToDigit (l : Letter) : Nat :=
  match l with
  | Letter.A => 0
  | Letter.B => 1
  | Letter.C => 2
  | Letter.D => 3
  | Letter.E => 4
  | Letter.X => 5
  | Letter.Y => 6
  | Letter.Z => 7

/-- Convert a word to its corresponding number in base 8 --/
def wordToNumber (w : Word) : Nat :=
  w.foldl (fun acc l => acc * 8 + letterToDigit l) 0

/-- The word CZYEB --/
def czyeb : Word := [Letter.C, Letter.Z, Letter.Y, Letter.E, Letter.B]

/-- The word XCEDA --/
def xceda : Word := [Letter.X, Letter.C, Letter.E, Letter.D, Letter.A]

/-- The theorem to be proved --/
theorem word_count_between_czyeb_and_xceda :
  (wordToNumber xceda) - (wordToNumber czyeb) - 1 = 9590 := by
  sorry

end word_count_between_czyeb_and_xceda_l4067_406756


namespace set_union_problem_l4067_406783

theorem set_union_problem (a b : ℝ) :
  let M : Set ℝ := {a, b}
  let N : Set ℝ := {a + 1, 3}
  M ∩ N = {2} →
  M ∪ N = {1, 2, 3} := by
sorry

end set_union_problem_l4067_406783


namespace parallel_line_through_point_l4067_406750

/-- A line parallel to y = 1/2x - 1 passing through (0, 3) has equation y = 1/2x + 3 -/
theorem parallel_line_through_point (k b : ℝ) : 
  (∀ x y : ℝ, y = k * x + b) →  -- The line has equation y = kx + b
  k = 1/2 →                    -- The line is parallel to y = 1/2x - 1
  3 = b →                      -- The line passes through (0, 3)
  ∀ x y : ℝ, y = 1/2 * x + 3   -- The equation of the line is y = 1/2x + 3
:= by sorry

end parallel_line_through_point_l4067_406750


namespace smallest_three_digit_pq2r_l4067_406754

theorem smallest_three_digit_pq2r : ∃ (p q r : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  126 = p * q^2 * r ∧
  (∀ (x p' q' r' : ℕ), 
    100 ≤ x ∧ x < 126 →
    Nat.Prime p' → Nat.Prime q' → Nat.Prime r' →
    p' ≠ q' → p' ≠ r' → q' ≠ r' →
    x ≠ p' * q'^2 * r') :=
by sorry

end smallest_three_digit_pq2r_l4067_406754


namespace solve_x_solve_y_solve_pqr_l4067_406719

-- Define the structure of the diagram for parts (a) and (b)
structure Diagram :=
  (top_left : ℤ)
  (top_right : ℤ)
  (bottom : ℤ)
  (top_sum : ℤ)
  (left_sum : ℤ)
  (right_sum : ℤ)

-- Define the diagram for part (a)
def diagram_a : Diagram :=
  { top_left := 9,  -- This is derived from the given information
    top_right := 4,
    bottom := 1,    -- This is derived from the given information
    top_sum := 13,
    left_sum := 10,
    right_sum := 5  -- This is x, which we need to prove
  }

-- Define the diagram for part (b)
def diagram_b : Diagram :=
  { top_left := 24,  -- This is 3w, where w = 8
    top_right := 24, -- This is also 3w
    bottom := 8,     -- This is w
    top_sum := 48,
    left_sum := 32,  -- This is y, which we need to prove
    right_sum := 32  -- This is also y
  }

-- Theorem for part (a)
theorem solve_x (d : Diagram) : 
  d.top_left + d.top_right = d.top_sum ∧
  d.top_left + d.bottom = d.left_sum ∧
  d.bottom + d.top_right = d.right_sum →
  d.right_sum = 5 :=
sorry

-- Theorem for part (b)
theorem solve_y (d : Diagram) :
  d.top_left = d.top_right ∧
  d.top_left = 3 * d.bottom ∧
  d.top_left + d.top_right = d.top_sum ∧
  d.top_left + d.bottom = d.left_sum ∧
  d.left_sum = d.right_sum →
  d.left_sum = 32 :=
sorry

-- Theorem for part (c)
theorem solve_pqr (p q r : ℤ) :
  p + r = 3 ∧
  p + q = 18 ∧
  q + r = 13 →
  p = 4 ∧ q = 14 ∧ r = -1 :=
sorry

end solve_x_solve_y_solve_pqr_l4067_406719


namespace whitney_max_sets_l4067_406791

/-- Represents the number of items Whitney has --/
structure ItemCounts where
  tshirts : ℕ
  buttons : ℕ
  stickers : ℕ
  keychains : ℕ

/-- Represents the requirements for each set --/
structure SetRequirements where
  tshirts : ℕ
  buttonToStickerRatio : ℕ
  keychains : ℕ

/-- Calculates the maximum number of sets that can be made --/
def maxSets (items : ItemCounts) (reqs : SetRequirements) : ℕ :=
  min (items.tshirts / reqs.tshirts)
    (min (items.buttons / reqs.buttonToStickerRatio)
      (min (items.stickers)
        (items.keychains / reqs.keychains)))

/-- Theorem stating that the maximum number of sets Whitney can make is 7 --/
theorem whitney_max_sets :
  let items := ItemCounts.mk 7 36 15 21
  let reqs := SetRequirements.mk 1 4 3
  maxSets items reqs = 7 := by
  sorry


end whitney_max_sets_l4067_406791


namespace football_throw_distance_l4067_406798

/-- Proves that Kyle threw the ball 24 yards farther than Parker -/
theorem football_throw_distance (parker_distance : ℝ) (grant_distance : ℝ) (kyle_distance : ℝ) :
  parker_distance = 16 ∧
  grant_distance = parker_distance * 1.25 ∧
  kyle_distance = grant_distance * 2 →
  kyle_distance - parker_distance = 24 := by
  sorry

end football_throw_distance_l4067_406798


namespace count_seven_100_to_199_l4067_406716

/-- Count of digit 7 in a number -/
def count_seven (n : ℕ) : ℕ := sorry

/-- Sum of count_seven for a range of numbers -/
def sum_count_seven (start finish : ℕ) : ℕ := sorry

theorem count_seven_100_to_199 :
  sum_count_seven 100 199 = 20 := by sorry

end count_seven_100_to_199_l4067_406716


namespace contrapositive_example_l4067_406728

theorem contrapositive_example : 
  (∀ x : ℝ, x > 2 → x > 1) ↔ (∀ x : ℝ, x ≤ 1 → x ≤ 2) := by
  sorry

end contrapositive_example_l4067_406728


namespace min_value_problem_l4067_406720

theorem min_value_problem (x y : ℝ) :
  (abs y ≤ 1) →
  (2 * x + y = 1) →
  (∀ x' y' : ℝ, abs y' ≤ 1 → 2 * x' + y' = 1 → 2 * x'^2 + 16 * x' + 3 * y'^2 ≥ 3) :=
by sorry

end min_value_problem_l4067_406720


namespace power_equality_l4067_406794

theorem power_equality (a b : ℝ) (h : (a - 2)^2 + |b + 1| = 0) : b^a = 1 := by
  sorry

end power_equality_l4067_406794


namespace wedge_volume_l4067_406736

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d : ℝ) (θ : ℝ) (h : d = 16 ∧ θ = 30 * π / 180) : 
  let r := d / 2
  let v := (r^2 * d * π) / 4
  v = 256 * π := by sorry

end wedge_volume_l4067_406736


namespace horner_method_proof_l4067_406711

def horner_polynomial (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def f (x : ℝ) : ℝ :=
  horner_polynomial [5, 4, 3, 2, 1, 0] x

theorem horner_method_proof :
  f 3 = 1641 := by
  sorry

end horner_method_proof_l4067_406711


namespace smallest_total_books_l4067_406745

/-- Represents the number of books for each subject -/
structure BookCounts where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Checks if the given book counts satisfy the required ratios -/
def satisfiesRatios (books : BookCounts) : Prop :=
  3 * books.chemistry = 2 * books.physics ∧
  4 * books.biology = 3 * books.chemistry

/-- Calculates the total number of books -/
def totalBooks (books : BookCounts) : ℕ :=
  books.physics + books.chemistry + books.biology

/-- Theorem stating the smallest possible total number of books -/
theorem smallest_total_books :
  ∃ (books : BookCounts),
    satisfiesRatios books ∧
    totalBooks books > 3000 ∧
    ∀ (other : BookCounts),
      satisfiesRatios other → totalBooks other > 3000 →
      totalBooks books ≤ totalBooks other :=
sorry

end smallest_total_books_l4067_406745


namespace rice_yield_comparison_l4067_406707

/-- Represents the rice field contract information -/
structure RiceContract where
  acres : ℕ
  yieldPerAcre : ℕ

/-- Calculates the total yield for a given contract -/
def totalYield (contract : RiceContract) : ℕ :=
  contract.acres * contract.yieldPerAcre

theorem rice_yield_comparison 
  (uncleLi : RiceContract)
  (auntLin : RiceContract)
  (h1 : uncleLi.acres = 12)
  (h2 : uncleLi.yieldPerAcre = 660)
  (h3 : auntLin.acres = uncleLi.acres - 2)
  (h4 : totalYield auntLin = totalYield uncleLi - 420) :
  totalYield uncleLi = 7920 ∧ 
  uncleLi.yieldPerAcre + 90 = auntLin.yieldPerAcre :=
by sorry

end rice_yield_comparison_l4067_406707


namespace mung_bean_germination_l4067_406739

theorem mung_bean_germination 
  (germination_rate : ℝ) 
  (total_seeds : ℝ) 
  (h1 : germination_rate = 0.971) 
  (h2 : total_seeds = 1000) : 
  total_seeds * (1 - germination_rate) = 29 := by
sorry

end mung_bean_germination_l4067_406739


namespace throwers_count_l4067_406795

/-- Represents a football team with throwers and non-throwers (left-handed and right-handed) -/
structure FootballTeam where
  total_players : ℕ
  throwers : ℕ
  left_handed : ℕ
  right_handed : ℕ

/-- Conditions for the football team -/
def valid_team (team : FootballTeam) : Prop :=
  team.total_players = 70 ∧
  team.throwers > 0 ∧
  team.throwers + team.left_handed + team.right_handed = team.total_players ∧
  team.left_handed = (team.total_players - team.throwers) / 3 ∧
  team.right_handed = team.throwers + 2 * team.left_handed ∧
  team.throwers + team.right_handed = 60

/-- Theorem stating that a valid team has 40 throwers -/
theorem throwers_count (team : FootballTeam) (h : valid_team team) : team.throwers = 40 := by
  sorry

end throwers_count_l4067_406795


namespace square_of_negative_product_l4067_406723

theorem square_of_negative_product (a b : ℝ) : (-a^2 * b)^2 = a^4 * b^2 := by
  sorry

end square_of_negative_product_l4067_406723


namespace unique_pair_l4067_406713

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 60 ∧ a < 70 ∧ b ≥ 60 ∧ b < 70 ∧ 
  a % 10 ≠ 6 ∧ b % 10 ≠ 6 ∧
  a * b = (10 * (a % 10) + 6) * (10 * (b % 10) + 6)

theorem unique_pair : 
  ∀ a b : ℕ, is_valid_pair a b → ((a = 69 ∧ b = 64) ∨ (a = 64 ∧ b = 69)) :=
by sorry

end unique_pair_l4067_406713


namespace tan_function_property_l4067_406704

noncomputable def f (a b x : ℝ) : ℝ := a * Real.tan (b * x)

theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, f a b (x + π/5) = f a b x) →
  f a b (5*π/24) = 5 →
  a * b = 25 / Real.tan (π/24) := by
sorry

end tan_function_property_l4067_406704


namespace acute_angles_trigonometry_l4067_406758

open Real

theorem acute_angles_trigonometry (α β : ℝ) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_tan_α : tan α = 2)
  (h_sin_diff : sin (α - β) = -sqrt 10 / 10) :
  sin (2 * α) = 4 / 5 ∧ tan (α + β) = -9 / 13 := by
sorry


end acute_angles_trigonometry_l4067_406758


namespace no_valid_score_l4067_406733

/-- Represents a player in the hockey match -/
inductive Player
| Anton
| Ilya
| Sergey

/-- Represents the statements made by each player -/
def Statement : Type := Player → ℕ

/-- The statements made by Anton -/
def AntonStatement : Statement :=
  fun p => match p with
  | Player.Anton => 3
  | Player.Ilya => 1
  | Player.Sergey => 0

/-- The statements made by Ilya -/
def IlyaStatement : Statement :=
  fun p => match p with
  | Player.Anton => 0
  | Player.Ilya => 4
  | Player.Sergey => 5

/-- The statements made by Sergey -/
def SergeyStatement : Statement :=
  fun p => match p with
  | Player.Anton => 2
  | Player.Ilya => 0
  | Player.Sergey => 6

/-- Checks if a given score satisfies the conditions -/
def satisfiesConditions (score : Player → ℕ) : Prop :=
  (score Player.Anton + score Player.Ilya + score Player.Sergey = 10) ∧
  (∃ (p : Player), AntonStatement p = score p) ∧
  (∃ (p : Player), AntonStatement p ≠ score p) ∧
  (∃ (p : Player), IlyaStatement p = score p) ∧
  (∃ (p : Player), IlyaStatement p ≠ score p) ∧
  (∃ (p : Player), SergeyStatement p = score p) ∧
  (∃ (p : Player), SergeyStatement p ≠ score p)

/-- Theorem stating that no score satisfies all conditions -/
theorem no_valid_score : ¬∃ (score : Player → ℕ), satisfiesConditions score := by
  sorry


end no_valid_score_l4067_406733


namespace expression_factorization_l4067_406782

theorem expression_factorization (b : ℝ) :
  (8 * b^3 + 120 * b^2 - 14) - (9 * b^3 - 2 * b^2 + 14) = -1 * (b^3 - 122 * b^2 + 28) := by
  sorry

end expression_factorization_l4067_406782


namespace complex_fraction_simplification_l4067_406797

theorem complex_fraction_simplification :
  (Complex.I + 3) / (Complex.I + 1) = 2 - Complex.I := by
  sorry

end complex_fraction_simplification_l4067_406797


namespace greatest_integer_inequality_l4067_406706

theorem greatest_integer_inequality : 
  (∃ (y : ℤ), (5 : ℚ) / 8 > (y : ℚ) / 15 ∧ 
    ∀ (z : ℤ), (5 : ℚ) / 8 > (z : ℚ) / 15 → z ≤ y) ∧ 
  (5 : ℚ) / 8 > (9 : ℚ) / 15 ∧ 
  (5 : ℚ) / 8 ≤ (10 : ℚ) / 15 :=
by sorry

end greatest_integer_inequality_l4067_406706


namespace log_equality_implies_ratio_l4067_406788

theorem log_equality_implies_ratio (p q : ℝ) (hp : 0 < p) (hq : 0 < q) :
  (Real.log p / Real.log 4 = Real.log q / Real.log 8) ∧
  (Real.log p / Real.log 4 = Real.log (p + q) / Real.log 18) →
  q / p = Real.sqrt p :=
by sorry

end log_equality_implies_ratio_l4067_406788


namespace hat_saves_greater_percentage_l4067_406701

-- Define the given values
def shoes_spent : ℚ := 42.25
def shoes_saved : ℚ := 3.75
def hat_sale_price : ℚ := 18.20
def hat_discount : ℚ := 1.80

-- Define the calculated values
def shoes_original : ℚ := shoes_spent + shoes_saved
def hat_original : ℚ := hat_sale_price + hat_discount

-- Define the percentage saved function
def percentage_saved (saved amount : ℚ) : ℚ := (saved / amount) * 100

-- Theorem statement
theorem hat_saves_greater_percentage :
  percentage_saved hat_discount hat_original > percentage_saved shoes_saved shoes_original :=
sorry

end hat_saves_greater_percentage_l4067_406701


namespace haploid_12_pairs_implies_tetraploid_l4067_406777

/-- Represents the ploidy level of a plant -/
inductive Ploidy
  | Diploid
  | Triploid
  | Tetraploid
  | Hexaploid

/-- Represents a potato plant -/
structure PotatoPlant where
  ploidy : Ploidy

/-- Represents a haploid plant derived from anther culture -/
structure HaploidPlant where
  chromosomePairs : Nat

/-- Function to determine the ploidy of the original plant based on the haploid plant's chromosome pairs -/
def determinePloidy (haploid : HaploidPlant) : Ploidy :=
  if haploid.chromosomePairs = 12 then Ploidy.Tetraploid else Ploidy.Diploid

/-- Theorem stating that if a haploid plant derived from anther culture forms 12 chromosome pairs,
    then the original potato plant is tetraploid -/
theorem haploid_12_pairs_implies_tetraploid (haploid : HaploidPlant) (original : PotatoPlant) :
  haploid.chromosomePairs = 12 → original.ploidy = Ploidy.Tetraploid :=
by
  sorry


end haploid_12_pairs_implies_tetraploid_l4067_406777


namespace q_polynomial_form_l4067_406743

-- Define q as a function from ℝ to ℝ
variable (q : ℝ → ℝ)

-- Define the theorem
theorem q_polynomial_form :
  (∀ x, q x + (x^6 + 4*x^4 + 8*x^2 + 7*x) = (12*x^4 + 30*x^3 + 40*x^2 + 10*x + 2)) →
  (∀ x, q x = -x^6 + 8*x^4 + 30*x^3 + 32*x^2 + 3*x + 2) :=
by
  sorry

end q_polynomial_form_l4067_406743


namespace tangent_line_to_parabola_l4067_406722

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero. -/
axiom tangent_iff_discriminant_zero (a b c : ℝ) :
  (∃ x y : ℝ, y = a*x + b ∧ y^2 = c*x) →
  (∀ x y : ℝ, y = a*x + b → y^2 = c*x → (a*x + b)^2 = c*x) →
  b^2 = a*c

/-- The main theorem: if y = 3x + c is tangent to y^2 = 12x, then c = 1 -/
theorem tangent_line_to_parabola (c : ℝ) :
  (∃ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x) →
  (∀ x y : ℝ, y = 3*x + c → y^2 = 12*x → (3*x + c)^2 = 12*x) →
  c = 1 := by
  sorry

end tangent_line_to_parabola_l4067_406722


namespace decreasing_linear_function_l4067_406729

theorem decreasing_linear_function (x1 x2 : ℝ) (h : x2 > x1) : -6 * x2 < -6 * x1 := by
  sorry

end decreasing_linear_function_l4067_406729


namespace multiplicative_inverse_201_mod_299_l4067_406724

theorem multiplicative_inverse_201_mod_299 :
  ∃! x : ℕ, x < 299 ∧ (201 * x) % 299 = 1 :=
by
  use 180
  sorry

end multiplicative_inverse_201_mod_299_l4067_406724


namespace average_side_length_of_squares_l4067_406718

theorem average_side_length_of_squares (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) : 
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 25 / 3 := by
  sorry

end average_side_length_of_squares_l4067_406718


namespace juniper_bones_theorem_l4067_406715

/-- Represents the number of bones Juniper has -/
def juniper_bones (initial : ℕ) (x : ℕ) (y : ℕ) : ℕ :=
  initial + x - y

theorem juniper_bones_theorem (x : ℕ) (y : ℕ) :
  juniper_bones 4 x y = 8 - y :=
by
  sorry

#check juniper_bones_theorem

end juniper_bones_theorem_l4067_406715


namespace triangle_ad_length_l4067_406748

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the perpendicular foot
def perpendicularFoot (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the ratio of line segments
def ratio (p q r s : ℝ × ℝ) : ℚ := sorry

theorem triangle_ad_length (abc : Triangle) :
  let A := abc.A
  let B := abc.B
  let C := abc.C
  let D := perpendicularFoot A B C
  length A B = 13 →
  length A C = 20 →
  ratio B D C D = 3/4 →
  length A D = 8 * Real.sqrt 2 := by sorry

end triangle_ad_length_l4067_406748


namespace charlie_extra_cost_l4067_406779

/-- Charlie's cell phone plan and usage details -/
structure CellPhonePlan where
  included_data : ℕ
  extra_cost_per_gb : ℕ
  week1_usage : ℕ
  week2_usage : ℕ
  week3_usage : ℕ
  week4_usage : ℕ

/-- Calculate the extra cost for Charlie's cell phone usage -/
def calculate_extra_cost (plan : CellPhonePlan) : ℕ :=
  let total_usage := plan.week1_usage + plan.week2_usage + plan.week3_usage + plan.week4_usage
  let over_limit := if total_usage > plan.included_data then total_usage - plan.included_data else 0
  over_limit * plan.extra_cost_per_gb

/-- Theorem: Charlie's extra cost is $120.00 -/
theorem charlie_extra_cost :
  let charlie_plan : CellPhonePlan := {
    included_data := 8,
    extra_cost_per_gb := 10,
    week1_usage := 2,
    week2_usage := 3,
    week3_usage := 5,
    week4_usage := 10
  }
  calculate_extra_cost charlie_plan = 120 := by
  sorry

end charlie_extra_cost_l4067_406779


namespace rightmost_three_digits_of_7_to_1993_l4067_406757

theorem rightmost_three_digits_of_7_to_1993 : 7^1993 % 1000 = 407 := by
  sorry

end rightmost_three_digits_of_7_to_1993_l4067_406757


namespace competition_participants_l4067_406769

theorem competition_participants (freshmen : ℕ) (sophomores : ℕ) : 
  freshmen = 8 → sophomores = 5 * freshmen → freshmen + sophomores = 48 := by
sorry

end competition_participants_l4067_406769


namespace initial_depth_is_40_l4067_406732

/-- Represents the work done by a group of workers digging to a certain depth -/
structure DiggingWork where
  workers : ℕ  -- number of workers
  hours   : ℕ  -- hours worked per day
  depth   : ℝ  -- depth dug in meters

/-- The theorem stating that given the initial and final conditions, the initial depth is 40 meters -/
theorem initial_depth_is_40 (initial final : DiggingWork) 
  (h1 : initial.workers = 45)
  (h2 : initial.hours = 8)
  (h3 : final.workers = initial.workers + 30)
  (h4 : final.hours = 6)
  (h5 : final.depth = 50)
  (h6 : initial.workers * initial.hours * initial.depth = final.workers * final.hours * final.depth) :
  initial.depth = 40 := by
  sorry

#check initial_depth_is_40

end initial_depth_is_40_l4067_406732


namespace slips_with_three_l4067_406761

/-- Given a bag with 15 slips, each having either 3 or 9, prove that if the expected value
    of a randomly drawn slip is 5, then 10 slips have 3 on them. -/
theorem slips_with_three (total : ℕ) (value_a value_b : ℕ) (expected : ℚ) : 
  total = 15 →
  value_a = 3 →
  value_b = 9 →
  expected = 5 →
  ∃ (count_a : ℕ), 
    count_a ≤ total ∧
    (count_a : ℚ) / total * value_a + (total - count_a : ℚ) / total * value_b = expected ∧
    count_a = 10 :=
by sorry

end slips_with_three_l4067_406761


namespace stock_price_decrease_l4067_406708

theorem stock_price_decrease (a : ℝ) (n : ℕ) (h₁ : a > 0) : a * (0.99 ^ n) < a := by
  sorry

end stock_price_decrease_l4067_406708


namespace seed_germination_probabilities_l4067_406751

/-- The number of seeds in each pit -/
def seeds_per_pit : ℕ := 3

/-- The probability of a single seed germinating -/
def germination_prob : ℝ := 0.5

/-- The number of pits -/
def num_pits : ℕ := 3

/-- The probability that at least one seed germinates in a pit -/
def prob_at_least_one_germinates : ℝ := 1 - (1 - germination_prob) ^ seeds_per_pit

/-- The probability that exactly two pits need replanting -/
def prob_exactly_two_need_replanting : ℝ := 
  (num_pits.choose 2) * (1 - prob_at_least_one_germinates) ^ 2 * prob_at_least_one_germinates

/-- The probability that at least one pit needs replanting -/
def prob_at_least_one_needs_replanting : ℝ := 1 - prob_at_least_one_germinates ^ num_pits

theorem seed_germination_probabilities :
  (prob_at_least_one_germinates = 0.875) ∧
  (prob_exactly_two_need_replanting = 0.713) ∧
  (prob_at_least_one_needs_replanting = 0.330) := by
  sorry

end seed_germination_probabilities_l4067_406751


namespace difference_of_squares_650_550_l4067_406781

theorem difference_of_squares_650_550 : 650^2 - 550^2 = 120000 := by
  sorry

end difference_of_squares_650_550_l4067_406781


namespace f_neither_odd_nor_even_l4067_406726

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2

-- Define the domain of f
def domain : Set ℝ := Set.Ioc (-5) 5

-- Theorem statement
theorem f_neither_odd_nor_even :
  ¬(∀ x ∈ domain, f x = -f (-x)) ∧ ¬(∀ x ∈ domain, f x = f (-x)) :=
sorry

end f_neither_odd_nor_even_l4067_406726


namespace students_not_picked_l4067_406773

theorem students_not_picked (total_students : ℕ) (num_groups : ℕ) (students_per_group : ℕ) 
  (h1 : total_students = 58)
  (h2 : num_groups = 8)
  (h3 : students_per_group = 6) : 
  total_students - (num_groups * students_per_group) = 10 := by
  sorry

end students_not_picked_l4067_406773


namespace simplest_form_l4067_406760

theorem simplest_form (a b : ℝ) (h : a ≠ b ∧ a ≠ -b) : 
  ¬∃ (f g : ℝ → ℝ → ℝ), ∀ (x y : ℝ), 
    (x^2 + y^2) / (x^2 - y^2) = f x y / g x y ∧ 
    (f x y ≠ x^2 + y^2 ∨ g x y ≠ x^2 - y^2) :=
sorry

end simplest_form_l4067_406760


namespace sum_cubes_minus_product_l4067_406762

theorem sum_cubes_minus_product (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 15)
  (sum_products_eq : a*b + a*c + a*d + b*c + b*d + c*d = 40) :
  a^3 + b^3 + c^3 + d^3 - 3*a*b*c*d = 1695 := by
  sorry

end sum_cubes_minus_product_l4067_406762


namespace workshop_analysis_l4067_406774

/-- Workshop worker information -/
structure Workshop where
  total_workers : ℕ
  avg_salary : ℝ
  technicians : ℕ
  technician_salary : ℝ
  managers : ℕ
  manager_salary : ℝ
  assistant_salary : ℝ

/-- Theorem about workshop workers and salaries -/
theorem workshop_analysis (w : Workshop)
  (h_total : w.total_workers = 20)
  (h_avg : w.avg_salary = 8000)
  (h_tech : w.technicians = 7)
  (h_tech_salary : w.technician_salary = 12000)
  (h_man : w.managers = 5)
  (h_man_salary : w.manager_salary = 15000)
  (h_assist_salary : w.assistant_salary = 6000) :
  let assistants := w.total_workers - w.technicians - w.managers
  let tech_man_total := w.technicians * w.technician_salary + w.managers * w.manager_salary
  let tech_man_avg := tech_man_total / (w.technicians + w.managers : ℝ)
  assistants = 8 ∧ tech_man_avg = 13250 := by
  sorry


end workshop_analysis_l4067_406774


namespace increasing_on_open_interval_l4067_406775

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Assume f' is the derivative of f
variable (h : ∀ x, HasDerivAt f (f' x) x)

-- Theorem statement
theorem increasing_on_open_interval
  (h1 : ∀ x ∈ Set.Ioo 4 5, f' x > 0) :
  StrictMonoOn f (Set.Ioo 4 5) :=
sorry

end increasing_on_open_interval_l4067_406775


namespace car_speed_time_relation_l4067_406768

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Theorem stating that if Car O travels at three times the speed of Car P for the same distance,
    then Car O's travel time is one-third of Car P's travel time -/
theorem car_speed_time_relation (p o : Car) (distance : ℝ) :
  o.speed = 3 * p.speed →
  distance = p.speed * p.time →
  distance = o.speed * o.time →
  o.time = p.time / 3 := by
  sorry


end car_speed_time_relation_l4067_406768


namespace arithmetic_calculation_l4067_406744

theorem arithmetic_calculation : 
  4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200 := by
  sorry

end arithmetic_calculation_l4067_406744


namespace rachels_pizza_consumption_l4067_406734

theorem rachels_pizza_consumption 
  (total_pizza : ℕ) 
  (bellas_pizza : ℕ) 
  (h1 : total_pizza = 952) 
  (h2 : bellas_pizza = 354) : 
  total_pizza - bellas_pizza = 598 := by
sorry

end rachels_pizza_consumption_l4067_406734


namespace next_term_is_512x4_l4067_406749

def geometric_sequence (x : ℝ) : ℕ → ℝ
  | 0 => 2
  | 1 => 8 * x
  | 2 => 32 * x^2
  | 3 => 128 * x^3
  | (n + 4) => geometric_sequence x n

theorem next_term_is_512x4 (x : ℝ) : geometric_sequence x 4 = 512 * x^4 := by
  sorry

end next_term_is_512x4_l4067_406749


namespace apple_grape_equivalence_l4067_406702

/-- If 3/4 of 12 apples are worth as much as 6 grapes, then 1/3 of 9 apples are worth as much as 2 grapes -/
theorem apple_grape_equivalence (apple_value grape_value : ℚ) : 
  (3 / 4 * 12 : ℚ) * apple_value = 6 * grape_value → 
  (1 / 3 * 9 : ℚ) * apple_value = 2 * grape_value := by
  sorry

end apple_grape_equivalence_l4067_406702


namespace correct_stratified_sampling_l4067_406763

/-- Represents the types of land in the farm --/
inductive LandType
  | Flat
  | Ditch
  | Sloped

/-- Represents the farm's land distribution --/
def farm : LandType → ℕ
  | LandType.Flat => 150
  | LandType.Ditch => 30
  | LandType.Sloped => 90

/-- Total acreage of the farm --/
def totalAcres : ℕ := farm LandType.Flat + farm LandType.Ditch + farm LandType.Sloped

/-- Sample size for the study --/
def sampleSize : ℕ := 18

/-- Calculates the sample size for each land type --/
def stratifiedSample (t : LandType) : ℕ :=
  (farm t * sampleSize) / totalAcres

/-- Theorem stating the correct stratified sampling for each land type --/
theorem correct_stratified_sampling :
  stratifiedSample LandType.Flat = 10 ∧
  stratifiedSample LandType.Ditch = 2 ∧
  stratifiedSample LandType.Sloped = 6 := by
  sorry


end correct_stratified_sampling_l4067_406763


namespace largest_even_number_less_than_150_div_9_l4067_406786

theorem largest_even_number_less_than_150_div_9 :
  ∃ (x : ℕ), 
    x % 2 = 0 ∧ 
    9 * x < 150 ∧ 
    ∀ (y : ℕ), y % 2 = 0 → 9 * y < 150 → y ≤ x ∧
    x = 16 :=
by sorry

end largest_even_number_less_than_150_div_9_l4067_406786


namespace optimal_plan_l4067_406735

/-- Represents the cost and quantity of new energy vehicles --/
structure VehiclePlan where
  costA : ℝ  -- Cost of A-type car in million yuan
  costB : ℝ  -- Cost of B-type car in million yuan
  quantA : ℕ -- Quantity of A-type cars
  quantB : ℕ -- Quantity of B-type cars

/-- Conditions for the vehicle purchase plan --/
def satisfiesConditions (plan : VehiclePlan) : Prop :=
  3 * plan.costA + plan.costB = 85 ∧
  2 * plan.costA + 4 * plan.costB = 140 ∧
  plan.quantA + plan.quantB = 15 ∧
  plan.quantA ≤ 2 * plan.quantB

/-- Total cost of the vehicle purchase plan --/
def totalCost (plan : VehiclePlan) : ℝ :=
  plan.costA * plan.quantA + plan.costB * plan.quantB

/-- Theorem stating the most cost-effective plan --/
theorem optimal_plan :
  ∃ (plan : VehiclePlan),
    satisfiesConditions plan ∧
    plan.costA = 20 ∧
    plan.costB = 25 ∧
    plan.quantA = 10 ∧
    plan.quantB = 5 ∧
    totalCost plan = 325 ∧
    (∀ (otherPlan : VehiclePlan),
      satisfiesConditions otherPlan →
      totalCost otherPlan ≥ totalCost plan) :=
by
  sorry


end optimal_plan_l4067_406735


namespace count_integers_eq_1278_l4067_406710

/-- Recursive function to calculate the number of n-digit sequences with no consecutive 1's -/
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => a (n + 1) + a n

/-- The number of 12-digit positive integers with all digits either 1 or 2 and exactly two consecutive 1's -/
def count_integers : ℕ := 2 * a 10 + 9 * (2 * a 9)

/-- Theorem stating that the count of such integers is 1278 -/
theorem count_integers_eq_1278 : count_integers = 1278 := by sorry

end count_integers_eq_1278_l4067_406710


namespace quadratic_root_shift_l4067_406766

theorem quadratic_root_shift (p q : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) ∧ (x₂^2 + p*x₂ + q = 0) →
  ((x₁ + 1)^2 + (p - 2)*(x₁ + 1) + (q - p + 1) = 0) ∧
  ((x₂ + 1)^2 + (p - 2)*(x₂ + 1) + (q - p + 1) = 0) := by
sorry

end quadratic_root_shift_l4067_406766


namespace cos_x_plus_pi_sixth_l4067_406789

theorem cos_x_plus_pi_sixth (x : ℝ) (h : Real.sin (π / 3 - x) = -3 / 5) : 
  Real.cos (x + π / 6) = -3 / 5 := by
sorry

end cos_x_plus_pi_sixth_l4067_406789


namespace cubic_function_max_value_l4067_406709

/-- Given a cubic function f with a known minimum value on an interval,
    prove that its maximum value on the same interval is 43. -/
theorem cubic_function_max_value (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^3 - 6 * x^2 + a
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f x ≤ f y) →
  (∃ x ∈ Set.Icc (-2) 2, f x = 3) →
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f y ≤ f x ∧ f x = 43) :=
by sorry

end cubic_function_max_value_l4067_406709


namespace min_teams_for_highest_score_fewer_wins_l4067_406714

/-- Represents a soccer team --/
structure Team :=
  (id : ℕ)
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)

/-- Calculates the score of a team --/
def score (t : Team) : ℕ := 2 * t.wins + t.draws

/-- Represents a soccer tournament --/
structure Tournament :=
  (teams : List Team)
  (numTeams : ℕ)
  (allPlayedAgainstEachOther : Bool)

/-- Checks if a team has the highest score in the tournament --/
def hasHighestScore (t : Team) (tournament : Tournament) : Prop :=
  ∀ other : Team, other ∈ tournament.teams → score t ≥ score other

/-- Checks if a team has fewer wins than all other teams --/
def hasFewerWins (t : Team) (tournament : Tournament) : Prop :=
  ∀ other : Team, other ∈ tournament.teams → other.id ≠ t.id → t.wins < other.wins

theorem min_teams_for_highest_score_fewer_wins (n : ℕ) :
  (∃ tournament : Tournament,
    tournament.numTeams = n ∧
    tournament.allPlayedAgainstEachOther = true ∧
    (∃ t : Team, t ∈ tournament.teams ∧ 
      hasHighestScore t tournament ∧
      hasFewerWins t tournament)) →
  n ≥ 6 :=
sorry

end min_teams_for_highest_score_fewer_wins_l4067_406714


namespace negation_of_universal_proposition_l4067_406741

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by sorry

end negation_of_universal_proposition_l4067_406741


namespace vector_ratio_theorem_l4067_406740

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_ratio_theorem (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : ‖a - b‖ = ‖a + 2 • b‖) 
  (h2 : inner a b / (‖a‖ * ‖b‖) = -1/4) : 
  ‖a‖ / ‖b‖ = 2 := by
  sorry

end vector_ratio_theorem_l4067_406740


namespace average_of_four_numbers_l4067_406703

theorem average_of_four_numbers (r s t u : ℝ) 
  (h : (5 / 4) * (r + s + t + u) = 15) : 
  (r + s + t + u) / 4 = 3 := by
  sorry

end average_of_four_numbers_l4067_406703


namespace equilateral_triangle_area_l4067_406796

/-- The area of an equilateral triangle, given specific internal perpendiculars -/
theorem equilateral_triangle_area (a b c : ℝ) (h : a = 2 ∧ b = 3 ∧ c = 4) : 
  ∃ (side : ℝ), 
    side > 0 ∧ 
    (a + b + c) * side / 2 = side * (side * Real.sqrt 3 / 2) / 2 ∧
    side * (side * Real.sqrt 3 / 2) / 2 = 27 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_area_l4067_406796


namespace matrix_commutator_similarity_l4067_406752

/-- Given n×n complex matrices A and B where A^2 = B^2, there exists an invertible n×n complex matrix S such that S(AB - BA) = (BA - AB)S. -/
theorem matrix_commutator_similarity {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℂ) 
  (h : A ^ 2 = B ^ 2) : 
  ∃ S : Matrix (Fin n) (Fin n) ℂ, IsUnit S ∧ S * (A * B - B * A) = (B * A - A * B) * S := by
  sorry

end matrix_commutator_similarity_l4067_406752


namespace skateboarder_distance_is_3720_l4067_406765

/-- Represents the skateboarder's journey -/
structure SkateboarderJourney where
  initial_distance : ℕ  -- Distance covered in the first second
  distance_increase : ℕ  -- Increase in distance each second on the ramp
  ramp_time : ℕ  -- Time spent on the ramp
  flat_time : ℕ  -- Time spent on the flat stretch

/-- Calculates the total distance traveled by the skateboarder -/
def total_distance (journey : SkateboarderJourney) : ℕ :=
  let ramp_distance := journey.ramp_time * (journey.initial_distance + (journey.ramp_time - 1) * journey.distance_increase / 2)
  let final_speed := journey.initial_distance + (journey.ramp_time - 1) * journey.distance_increase
  let flat_distance := final_speed * journey.flat_time
  ramp_distance + flat_distance

/-- Theorem stating that the total distance traveled is 3720 meters -/
theorem skateboarder_distance_is_3720 (journey : SkateboarderJourney) 
  (h1 : journey.initial_distance = 10)
  (h2 : journey.distance_increase = 9)
  (h3 : journey.ramp_time = 20)
  (h4 : journey.flat_time = 10) : 
  total_distance journey = 3720 := by
  sorry

end skateboarder_distance_is_3720_l4067_406765


namespace probability_of_white_ball_l4067_406790

-- Define the number of red and white balls
def num_red_balls : ℕ := 3
def num_white_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_white_balls

-- Define the probability of drawing a white ball
def prob_white_ball : ℚ := num_white_balls / total_balls

-- Theorem statement
theorem probability_of_white_ball :
  prob_white_ball = 2 / 5 := by sorry

end probability_of_white_ball_l4067_406790


namespace shoe_promotion_savings_difference_l4067_406717

/-- Calculates the savings difference between two promotions for shoe purchases -/
theorem shoe_promotion_savings_difference : 
  let original_price : ℝ := 50
  let promotion_c_discount : ℝ := 0.20
  let promotion_d_discount : ℝ := 15
  let cost_c : ℝ := original_price + (original_price * (1 - promotion_c_discount))
  let cost_d : ℝ := original_price + (original_price - promotion_d_discount)
  cost_c - cost_d = 5 := by sorry

end shoe_promotion_savings_difference_l4067_406717


namespace cos_alpha_terminal_point_l4067_406778

/-- Given a point P(-12, 5) on the terminal side of angle α, prove that cos α = -12/13 -/
theorem cos_alpha_terminal_point (α : Real) :
  let P : Real × Real := (-12, 5)
  (P.1 = -12 ∧ P.2 = 5) → -- Point P is (-12, 5)
  (P.1 = -12 * Real.cos α ∧ P.2 = -12 * Real.sin α) → -- P is on the terminal side of α
  Real.cos α = -12/13 := by
sorry

end cos_alpha_terminal_point_l4067_406778


namespace farmer_cows_distribution_l4067_406767

theorem farmer_cows_distribution (total : ℕ) : 
  (total : ℚ) / 3 + (total : ℚ) / 6 + (total : ℚ) / 8 + 15 = total → total = 40 := by
  sorry

end farmer_cows_distribution_l4067_406767


namespace complex_fraction_real_l4067_406787

theorem complex_fraction_real (m : ℝ) : 
  (((1 : ℂ) + m * Complex.I) / ((1 : ℂ) + Complex.I)).im = 0 → m = 1 := by
  sorry

end complex_fraction_real_l4067_406787


namespace video_game_lives_l4067_406792

theorem video_game_lives (initial_lives next_level_lives total_lives : ℝ) 
  (h1 : initial_lives = 43.0)
  (h2 : next_level_lives = 27.0)
  (h3 : total_lives = 84) :
  ∃ hard_part_lives : ℝ, 
    hard_part_lives = 14.0 ∧ 
    initial_lives + hard_part_lives + next_level_lives = total_lives :=
by sorry

end video_game_lives_l4067_406792


namespace arithmetic_equality_l4067_406727

theorem arithmetic_equality : 202 - 101 + 9 = 110 := by
  sorry

end arithmetic_equality_l4067_406727


namespace quadratic_minimum_l4067_406721

/-- The quadratic function f(x) = 3x^2 - 8x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 8 * x + 7

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6 * x - 8

theorem quadratic_minimum :
  ∃ (x_min : ℝ), x_min = 4/3 ∧ ∀ (x : ℝ), f x ≥ f x_min :=
by
  sorry


end quadratic_minimum_l4067_406721


namespace opposite_of_negative_two_l4067_406705

theorem opposite_of_negative_two : -(-(2 : ℤ)) = 2 := by sorry

end opposite_of_negative_two_l4067_406705


namespace expected_abs_difference_10_days_l4067_406776

/-- Represents the outcome of a single day --/
inductive DailyOutcome
| CatWins
| FoxWins
| BothLose

/-- Probability distribution for daily outcomes --/
def dailyProbability (outcome : DailyOutcome) : ℝ :=
  match outcome with
  | DailyOutcome.CatWins => 0.25
  | DailyOutcome.FoxWins => 0.25
  | DailyOutcome.BothLose => 0.5

/-- Expected value of the absolute difference in wealth after n days --/
def expectedAbsDifference (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating the expected absolute difference after 10 days is 1 --/
theorem expected_abs_difference_10_days :
  expectedAbsDifference 10 = 1 :=
sorry

end expected_abs_difference_10_days_l4067_406776


namespace b_and_d_know_grades_l4067_406737

-- Define the grade types
inductive Grade
| Excellent
| Good

-- Define the students
inductive Student
| A
| B
| C
| D

-- Function to represent the actual grade of a student
def actualGrade : Student → Grade := sorry

-- Function to represent what grades a student can see
def canSee : Student → Student → Prop := sorry

-- Theorem statement
theorem b_and_d_know_grades :
  -- There are 2 excellent grades and 2 good grades
  (∃ (s1 s2 s3 s4 : Student), s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4 ∧
    actualGrade s1 = Grade.Excellent ∧ actualGrade s2 = Grade.Excellent ∧
    actualGrade s3 = Grade.Good ∧ actualGrade s4 = Grade.Good) →
  -- A, B, and C can see each other's grades
  (canSee Student.A Student.B ∧ canSee Student.A Student.C ∧
   canSee Student.B Student.A ∧ canSee Student.B Student.C ∧
   canSee Student.C Student.A ∧ canSee Student.C Student.B) →
  -- B and C can see each other's grades
  (canSee Student.B Student.C ∧ canSee Student.C Student.B) →
  -- D and A can see each other's grades
  (canSee Student.D Student.A ∧ canSee Student.A Student.D) →
  -- A doesn't know their own grade after seeing B and C's grades
  (∃ (g1 g2 : Grade), g1 ≠ g2 ∧
    ((actualGrade Student.B = g1 ∧ actualGrade Student.C = g2) ∨
     (actualGrade Student.B = g2 ∧ actualGrade Student.C = g1))) →
  -- B and D can know their own grades
  (∃ (gb gd : Grade),
    (actualGrade Student.B = gb ∧ ∀ g, actualGrade Student.B = g → g = gb) ∧
    (actualGrade Student.D = gd ∧ ∀ g, actualGrade Student.D = g → g = gd))
  := by sorry

end b_and_d_know_grades_l4067_406737


namespace find_g_value_l4067_406712

theorem find_g_value (x g : ℝ) (h1 : x = 0.3) 
  (h2 : (10 * x + 2) / 4 - (3 * x - 6) / 18 = (g * x + 4) / 3) : g = 2 := by
  sorry

end find_g_value_l4067_406712


namespace min_sum_of_reciprocal_equation_l4067_406730

theorem min_sum_of_reciprocal_equation : 
  ∃ (x y z : ℕ+), 
    (1 : ℝ) / x + 4 / y + 9 / z = 1 ∧ 
    x + y + z = 36 ∧ 
    ∀ (a b c : ℕ+), (1 : ℝ) / a + 4 / b + 9 / c = 1 → a + b + c ≥ 36 := by
  sorry

end min_sum_of_reciprocal_equation_l4067_406730


namespace hall_people_count_l4067_406785

theorem hall_people_count (total_desks : ℕ) (occupied_desks : ℕ) (people : ℕ) : 
  total_desks = 72 →
  occupied_desks = 60 →
  people * 4 = occupied_desks * 5 →
  total_desks - occupied_desks = 12 →
  people = 75 := by
sorry

end hall_people_count_l4067_406785


namespace exponent_multiplication_l4067_406772

theorem exponent_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end exponent_multiplication_l4067_406772


namespace triangle_side_length_l4067_406731

/-- Represents a triangle with given side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.side1 + t.side2 + t.side3

theorem triangle_side_length 
  (t : Triangle) 
  (h_perimeter : t.perimeter = 160) 
  (h_side1 : t.side1 = 40) 
  (h_side3 : t.side3 = 70) : 
  t.side2 = 50 := by
sorry

end triangle_side_length_l4067_406731


namespace cubic_equation_solution_l4067_406793

theorem cubic_equation_solution :
  ∃! x : ℝ, (2010 + x)^3 = -x^3 ∧ x = -1005 := by sorry

end cubic_equation_solution_l4067_406793


namespace stating_max_fully_connected_is_N_minus_1_l4067_406799

/-- Represents a network of computers. -/
structure Network where
  N : ℕ
  not_fully_connected : ∃ (node : Fin N), ∃ (other : Fin N), node ≠ other
  N_gt_3 : N > 3

/-- The maximum number of fully connected nodes in the network. -/
def max_fully_connected (net : Network) : ℕ := net.N - 1

/-- 
Theorem stating that the maximum number of fully connected nodes 
in a network with the given conditions is N-1.
-/
theorem max_fully_connected_is_N_minus_1 (net : Network) : 
  max_fully_connected net = net.N - 1 := by
  sorry


end stating_max_fully_connected_is_N_minus_1_l4067_406799
