import Mathlib

namespace NUMINAMATH_CALUDE_apple_relationship_l395_39567

/-- Proves the relationship between bruised and wormy apples --/
theorem apple_relationship (total_apples wormy_ratio raw_apples : ℕ) 
  (h1 : total_apples = 85)
  (h2 : wormy_ratio = 5)
  (h3 : raw_apples = 42) : 
  ∃ (bruised wormy : ℕ), 
    wormy = total_apples / wormy_ratio ∧ 
    bruised = total_apples - raw_apples - wormy ∧ 
    bruised = wormy + 9 :=
by sorry

end NUMINAMATH_CALUDE_apple_relationship_l395_39567


namespace NUMINAMATH_CALUDE_product_of_number_and_sum_of_digits_l395_39526

theorem product_of_number_and_sum_of_digits : 
  let n : ℕ := 26
  let tens : ℕ := n / 10
  let units : ℕ := n % 10
  units = tens + 4 →
  n * (tens + units) = 208 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_number_and_sum_of_digits_l395_39526


namespace NUMINAMATH_CALUDE_coat_value_problem_l395_39599

/-- Represents the problem of determining the value of a coat given to a worker --/
theorem coat_value_problem (total_pay : ℝ) (yearly_cash : ℝ) (months_worked : ℝ) 
  (partial_cash : ℝ) (h1 : total_pay = yearly_cash + coat_value) 
  (h2 : yearly_cash = 12) (h3 : months_worked = 7) (h4 : partial_cash = 5) :
  ∃ coat_value : ℝ, coat_value = 4.8 ∧ 
    (months_worked / 12) * total_pay = partial_cash + coat_value := by
  sorry


end NUMINAMATH_CALUDE_coat_value_problem_l395_39599


namespace NUMINAMATH_CALUDE_simplify_A_minus_B_A_minus_B_value_l395_39540

/-- Given two real numbers a and b, we define A and B as follows -/
def A (a b : ℝ) : ℝ := (a + b)^2 - 3 * b^2

def B (a b : ℝ) : ℝ := 2 * (a + b) * (a - b) - 3 * a * b

/-- Theorem stating that A - B simplifies to -a^2 + 5ab -/
theorem simplify_A_minus_B (a b : ℝ) : A a b - B a b = -a^2 + 5*a*b := by sorry

/-- Theorem stating that if (a-3)^2 + |b-4| = 0, then A - B = 51 -/
theorem A_minus_B_value (a b : ℝ) (h : (a - 3)^2 + |b - 4| = 0) : A a b - B a b = 51 := by sorry

end NUMINAMATH_CALUDE_simplify_A_minus_B_A_minus_B_value_l395_39540


namespace NUMINAMATH_CALUDE_total_trees_planted_l395_39508

def trees_planted (fourth_grade fifth_grade sixth_grade : ℕ) : Prop :=
  fourth_grade = 30 ∧
  fifth_grade = 2 * fourth_grade ∧
  sixth_grade = 3 * fifth_grade - 30

theorem total_trees_planted :
  ∀ fourth_grade fifth_grade sixth_grade : ℕ,
  trees_planted fourth_grade fifth_grade sixth_grade →
  fourth_grade + fifth_grade + sixth_grade = 240 :=
by sorry

end NUMINAMATH_CALUDE_total_trees_planted_l395_39508


namespace NUMINAMATH_CALUDE_birds_taken_out_l395_39550

theorem birds_taken_out (initial_birds remaining_birds : ℕ) 
  (h1 : initial_birds = 19)
  (h2 : remaining_birds = 9) :
  initial_birds - remaining_birds = 10 := by
  sorry

end NUMINAMATH_CALUDE_birds_taken_out_l395_39550


namespace NUMINAMATH_CALUDE_club_members_proof_l395_39585

theorem club_members_proof (total : ℕ) (left_handed : ℕ) (jazz_lovers : ℕ) (right_handed_jazz_dislikers : ℕ) 
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : jazz_lovers = 18)
  (h4 : right_handed_jazz_dislikers = 2)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ x : ℕ, x = 5 ∧ 
    x + (left_handed - x) + (jazz_lovers - x) + right_handed_jazz_dislikers = total :=
by sorry

end NUMINAMATH_CALUDE_club_members_proof_l395_39585


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l395_39551

-- Problem 1
theorem problem_1 (x : ℝ) (h : x > 0) (eq : Real.sqrt x + 1 / Real.sqrt x = 3) : 
  x + 1 / x = 7 := by sorry

-- Problem 2
theorem problem_2 : 
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * (Real.log 2 / Real.log 3 + Real.log 8 / Real.log 9) = 25 / 12 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l395_39551


namespace NUMINAMATH_CALUDE_div_exp_eq_pow_specific_calculation_l395_39522

/-- Division exponentiation for rational numbers -/
def div_exp (a : ℚ) (n : ℕ) : ℚ :=
  if n ≤ 1 then a else (1 / a) ^ (n - 2)

/-- Theorem for division exponentiation -/
theorem div_exp_eq_pow (a : ℚ) (n : ℕ) (h : a ≠ 0) :
  div_exp a n = (1 / a) ^ (n - 2) :=
sorry

/-- Theorem for specific calculation -/
theorem specific_calculation :
  2^2 * div_exp (-1/3) 4 / div_exp (-2) 3 - div_exp (-3) 2 = -73 :=
sorry

end NUMINAMATH_CALUDE_div_exp_eq_pow_specific_calculation_l395_39522


namespace NUMINAMATH_CALUDE_inequality_implication_l395_39506

theorem inequality_implication (a b c d e : ℝ) :
  a * b^2 * c^3 * d^4 * e^5 < 0 → a * b^2 * c * d^4 * e < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l395_39506


namespace NUMINAMATH_CALUDE_stamp_problem_l395_39546

/-- Given stamps of denominations 5, n, and n+1 cents, 
    where n is a positive integer, 
    if 97 cents is the greatest postage that cannot be formed, 
    then n = 25 -/
theorem stamp_problem (n : ℕ) : 
  n > 0 → 
  (∀ k : ℕ, k > 97 → ∃ a b c : ℕ, k = 5*a + n*b + (n+1)*c) → 
  (∃ a b c : ℕ, 97 = 5*a + n*b + (n+1)*c → False) → 
  n = 25 := by
  sorry

end NUMINAMATH_CALUDE_stamp_problem_l395_39546


namespace NUMINAMATH_CALUDE_allocation_theorem_l395_39501

/-- The number of ways to allocate employees to departments -/
def allocation_count (total_employees : ℕ) (num_departments : ℕ) : ℕ :=
  sorry

/-- Two employees are considered as one unit -/
def combined_employees : ℕ := 4

/-- Number of ways to distribute combined employees into departments -/
def distribution_ways : ℕ := sorry

/-- Number of ways to assign groups to departments -/
def assignment_ways : ℕ := sorry

theorem allocation_theorem :
  allocation_count 5 3 = distribution_ways * assignment_ways ∧
  distribution_ways = 6 ∧
  assignment_ways = 6 ∧
  allocation_count 5 3 = 36 :=
sorry

end NUMINAMATH_CALUDE_allocation_theorem_l395_39501


namespace NUMINAMATH_CALUDE_galaxy_gym_member_ratio_l395_39524

theorem galaxy_gym_member_ratio :
  ∀ (f m : ℕ) (f_avg m_avg total_avg : ℝ),
    f_avg = 35 →
    m_avg = 45 →
    total_avg = 40 →
    (f_avg * f + m_avg * m) / (f + m) = total_avg →
    f = m :=
by
  sorry

end NUMINAMATH_CALUDE_galaxy_gym_member_ratio_l395_39524


namespace NUMINAMATH_CALUDE_child_play_time_l395_39565

theorem child_play_time (num_children : ℕ) (children_per_game : ℕ) (total_time : ℕ) 
  (h1 : num_children = 7)
  (h2 : children_per_game = 2)
  (h3 : total_time = 140)
  (h4 : children_per_game ≤ num_children)
  (h5 : children_per_game > 0)
  (h6 : total_time > 0) :
  (children_per_game * total_time) / num_children = 40 := by
sorry

end NUMINAMATH_CALUDE_child_play_time_l395_39565


namespace NUMINAMATH_CALUDE_luisa_pet_store_distance_l395_39541

theorem luisa_pet_store_distance (grocery_store_distance : ℝ) (mall_distance : ℝ) (home_distance : ℝ) 
  (miles_per_gallon : ℝ) (cost_per_gallon : ℝ) (total_cost : ℝ) :
  grocery_store_distance = 10 →
  mall_distance = 6 →
  home_distance = 9 →
  miles_per_gallon = 15 →
  cost_per_gallon = 3.5 →
  total_cost = 7 →
  ∃ (pet_store_distance : ℝ),
    pet_store_distance = 5 ∧
    grocery_store_distance + mall_distance + pet_store_distance + home_distance = 
      (total_cost / cost_per_gallon) * miles_per_gallon :=
by sorry

end NUMINAMATH_CALUDE_luisa_pet_store_distance_l395_39541


namespace NUMINAMATH_CALUDE_prob_green_ball_is_five_ninths_l395_39552

structure Container where
  red : ℕ
  green : ℕ

def containers : List Container := [
  ⟨10, 5⟩,
  ⟨3, 6⟩,
  ⟨4, 8⟩
]

def total_balls (c : Container) : ℕ := c.red + c.green

def prob_green (c : Container) : ℚ :=
  c.green / (total_balls c)

theorem prob_green_ball_is_five_ninths :
  (List.sum (containers.map (λ c => (1 : ℚ) / containers.length * prob_green c))) = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_ball_is_five_ninths_l395_39552


namespace NUMINAMATH_CALUDE_folding_crease_set_l395_39577

/-- Given a circle with center O(0,0) and radius R, and a point A(a,0) inside the circle,
    the set of all points P(x,y) that are equidistant from A and any point A' on the circumference
    of the circle satisfies the given inequality. -/
theorem folding_crease_set (R a x y : ℝ) (h1 : R > 0) (h2 : 0 < a ∧ a < R) :
  (x - a/2)^2 / (R/2)^2 + y^2 / ((R/2)^2 - (a/2)^2) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_folding_crease_set_l395_39577


namespace NUMINAMATH_CALUDE_square_side_length_l395_39504

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents an octagon -/
structure Octagon :=
  (A B C D E F G H : Point)

/-- Represents a square -/
structure Square :=
  (W X Y Z : Point)

/-- The octagon ABCDEFGH -/
def octagon : Octagon := sorry

/-- The inscribed square WXYZ -/
def square : Square := sorry

/-- W is on BC -/
axiom W_on_BC : square.W.x ≥ octagon.B.x ∧ square.W.x ≤ octagon.C.x ∧ 
                square.W.y = octagon.B.y ∧ square.W.y = octagon.C.y

/-- X is on DE -/
axiom X_on_DE : square.X.x ≥ octagon.D.x ∧ square.X.x ≤ octagon.E.x ∧ 
                square.X.y = octagon.D.y ∧ square.X.y = octagon.E.y

/-- Y is on FG -/
axiom Y_on_FG : square.Y.x ≥ octagon.F.x ∧ square.Y.x ≤ octagon.G.x ∧ 
                square.Y.y = octagon.F.y ∧ square.Y.y = octagon.G.y

/-- Z is on HA -/
axiom Z_on_HA : square.Z.x ≥ octagon.H.x ∧ square.Z.x ≤ octagon.A.x ∧ 
                square.Z.y = octagon.H.y ∧ square.Z.y = octagon.A.y

/-- AB = 50 -/
axiom AB_length : Real.sqrt ((octagon.A.x - octagon.B.x)^2 + (octagon.A.y - octagon.B.y)^2) = 50

/-- GH = 50(√3 - 1) -/
axiom GH_length : Real.sqrt ((octagon.G.x - octagon.H.x)^2 + (octagon.G.y - octagon.H.y)^2) = 50 * (Real.sqrt 3 - 1)

/-- The side length of square WXYZ is 50 -/
theorem square_side_length : 
  Real.sqrt ((square.W.x - square.Z.x)^2 + (square.W.y - square.Z.y)^2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l395_39504


namespace NUMINAMATH_CALUDE_star_equation_solution_l395_39587

def star (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

theorem star_equation_solution :
  ∀ A : ℝ, star A 6 = 31 → A = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l395_39587


namespace NUMINAMATH_CALUDE_vertex_angle_of_special_triangle_l395_39527

/-- A triangle with angles a, b, and c is isosceles and a "double angle triangle" -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_180 : a + b + c = 180
  isosceles : b = c
  double_angle : a = 2 * b ∨ b = 2 * a

/-- The vertex angle of an isosceles "double angle triangle" is either 36° or 90° -/
theorem vertex_angle_of_special_triangle (t : SpecialTriangle) :
  t.a = 36 ∨ t.a = 90 := by
  sorry

end NUMINAMATH_CALUDE_vertex_angle_of_special_triangle_l395_39527


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l395_39514

theorem triangle_side_ratio (a b c k : ℝ) 
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_ratio : a^2 + b^2 = k * c^2) : 
  k > 0.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l395_39514


namespace NUMINAMATH_CALUDE_orange_harvest_per_day_l395_39510

theorem orange_harvest_per_day (total_sacks : ℕ) (total_days : ℕ) (sacks_per_day : ℕ) : 
  total_sacks = 56 → total_days = 4 → sacks_per_day = total_sacks / total_days → sacks_per_day = 14 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_per_day_l395_39510


namespace NUMINAMATH_CALUDE_tg_ctg_roots_relation_l395_39554

-- Define the tangent and cotangent functions
noncomputable def tg (α : Real) : Real := Real.tan α
noncomputable def ctg (α : Real) : Real := 1 / Real.tan α

-- State the theorem
theorem tg_ctg_roots_relation (p q r s α β : Real) :
  (tg α)^2 - p * (tg α) + q = 0 ∧
  (tg β)^2 - p * (tg β) + q = 0 ∧
  (ctg α)^2 - r * (ctg α) + s = 0 ∧
  (ctg β)^2 - r * (ctg β) + s = 0 →
  r * s = p / q^2 := by
sorry

end NUMINAMATH_CALUDE_tg_ctg_roots_relation_l395_39554


namespace NUMINAMATH_CALUDE_graph_relationship_l395_39523

theorem graph_relationship (x : ℝ) : |x^2 - 3/2*x + 3| ≥ x^2 + 3/2*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_graph_relationship_l395_39523


namespace NUMINAMATH_CALUDE_five_dice_probability_l395_39598

/-- The probability of rolling a number greater than 1 on a single die -/
def prob_not_one : ℚ := 5/6

/-- The number of ways to choose 2 dice out of 5 -/
def choose_two_from_five : ℕ := 10

/-- The probability of two dice summing to 10 -/
def prob_sum_ten : ℚ := 1/12

/-- The probability of rolling five dice where none show 1 and two of them sum to 10 -/
def prob_five_dice : ℚ := (prob_not_one ^ 5) * choose_two_from_five * prob_sum_ten

theorem five_dice_probability :
  prob_five_dice = 2604.1667 / 7776 :=
sorry

end NUMINAMATH_CALUDE_five_dice_probability_l395_39598


namespace NUMINAMATH_CALUDE_triangle_special_angle_l395_39592

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a^2 + b^2 = c^2 - √2ab, then angle C = 3π/4 -/
theorem triangle_special_angle (a b c : ℝ) (h : a^2 + b^2 = c^2 - Real.sqrt 2 * a * b) :
  let angle_C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  angle_C = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l395_39592


namespace NUMINAMATH_CALUDE_total_rainfall_l395_39575

def rainfall_problem (first_week : ℝ) (second_week : ℝ) : Prop :=
  (second_week = 1.5 * first_week) ∧
  (second_week = 18) ∧
  (first_week + second_week = 30)

theorem total_rainfall : ∃ (first_week second_week : ℝ),
  rainfall_problem first_week second_week :=
by sorry

end NUMINAMATH_CALUDE_total_rainfall_l395_39575


namespace NUMINAMATH_CALUDE_contour_bar_chart_judges_relationship_l395_39562

/-- Represents a method for judging the relationship between categorical variables -/
inductive IndependenceTestMethod
  | Residuals
  | ContourBarChart
  | HypothesisTesting
  | Other

/-- Defines the property of being able to roughly judge the relationship between categorical variables -/
def can_roughly_judge_relationship (method : IndependenceTestMethod) : Prop :=
  match method with
  | IndependenceTestMethod.ContourBarChart => True
  | _ => False

/-- Theorem stating that a contour bar chart can be used to roughly judge the relationship between categorical variables in an independence test -/
theorem contour_bar_chart_judges_relationship :
  can_roughly_judge_relationship IndependenceTestMethod.ContourBarChart :=
sorry

end NUMINAMATH_CALUDE_contour_bar_chart_judges_relationship_l395_39562


namespace NUMINAMATH_CALUDE_folded_paper_thickness_l395_39511

/-- The thickness of a folded paper stack -/
def folded_thickness (initial_thickness : ℝ) : ℝ := 2 * initial_thickness

/-- Theorem: Folding a 0.2 cm thick paper stack once results in a 0.4 cm thick stack -/
theorem folded_paper_thickness :
  folded_thickness 0.2 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_thickness_l395_39511


namespace NUMINAMATH_CALUDE_remainder_sum_equality_l395_39513

/-- The remainder sum function -/
def r (n : ℕ) : ℕ := (Finset.range n).sum (λ i => n % (i + 1))

/-- Theorem: The remainder sum of 2^k - 1 equals the remainder sum of 2^k for all k ≥ 1 -/
theorem remainder_sum_equality (k : ℕ) (hk : k ≥ 1) : r (2^k - 1) = r (2^k) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_equality_l395_39513


namespace NUMINAMATH_CALUDE_equilateral_triangle_condition_l395_39590

/-- A function that checks if a natural number n satisfies the conditions for forming an equilateral triangle with sticks of lengths 1 to n. -/
def can_form_equilateral_triangle (n : ℕ) : Prop :=
  n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5)

/-- The sum of the first n natural numbers. -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating that sticks of lengths 1 to n can form an equilateral triangle
    if and only if n satisfies the specific conditions. -/
theorem equilateral_triangle_condition (n : ℕ) :
  (sum_first_n n % 3 = 0 ∧ ∀ k < n, k > 0) ↔ can_form_equilateral_triangle n := by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_condition_l395_39590


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_l395_39502

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units -/
theorem circumscribed_circle_area (s : ℝ) (h : s = 12) : 
  let R := s / Real.sqrt 3
  π * R^2 = 48 * π := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_l395_39502


namespace NUMINAMATH_CALUDE_max_sum_with_length_constraint_l395_39547

-- Define the length function
def length (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem max_sum_with_length_constraint :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ length x + length y = 16 ∧ 
  ∀ (a b : ℕ), a > 1 → b > 1 → length a + length b = 16 → 
  a + 3 * b ≤ x + 3 * y ∧ x + 3 * y = 98306 :=
sorry

end NUMINAMATH_CALUDE_max_sum_with_length_constraint_l395_39547


namespace NUMINAMATH_CALUDE_tony_temperature_l395_39542

/-- Represents the temperature change caused by an illness -/
structure Illness where
  temp_change : Int

/-- Calculates the final temperature and its relation to the fever threshold -/
def calculate_temperature (normal_temp : Int) (illnesses : List Illness) (fever_threshold : Int) :
  (Int × Int) :=
  let final_temp := normal_temp + (illnesses.map (·.temp_change)).sum
  let above_threshold := final_temp - fever_threshold
  (final_temp, above_threshold)

theorem tony_temperature :
  let normal_temp := 95
  let illness_a := Illness.mk 10
  let illness_b := Illness.mk 4
  let illness_c := Illness.mk (-2)
  let illnesses := [illness_a, illness_b, illness_c]
  let fever_threshold := 100
  calculate_temperature normal_temp illnesses fever_threshold = (107, 7) := by
  sorry

end NUMINAMATH_CALUDE_tony_temperature_l395_39542


namespace NUMINAMATH_CALUDE_second_book_length_is_100_l395_39521

/-- The length of Yasna's first book in pages -/
def first_book_length : ℕ := 180

/-- The number of pages Yasna reads per day -/
def pages_per_day : ℕ := 20

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- The total number of pages Yasna reads in two weeks -/
def total_pages : ℕ := pages_per_day * days_in_two_weeks

/-- The length of Yasna's second book in pages -/
def second_book_length : ℕ := total_pages - first_book_length

theorem second_book_length_is_100 : second_book_length = 100 := by
  sorry

end NUMINAMATH_CALUDE_second_book_length_is_100_l395_39521


namespace NUMINAMATH_CALUDE_present_age_of_B_l395_39532

/-- Given two natural numbers A and B representing ages, proves that B is 41 years old
    given the conditions:
    1) In 10 years, A will be twice as old as B was 10 years ago.
    2) A is now 11 years older than B. -/
theorem present_age_of_B (A B : ℕ) 
    (h1 : A + 10 = 2 * (B - 10))
    (h2 : A = B + 11) : 
  B = 41 := by
  sorry


end NUMINAMATH_CALUDE_present_age_of_B_l395_39532


namespace NUMINAMATH_CALUDE_square_roots_problem_l395_39559

theorem square_roots_problem (a : ℝ) (x : ℝ) (h1 : a > 0) 
  (h2 : (2*x - 3)^2 = a) (h3 : (5 - x)^2 = a) : a = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l395_39559


namespace NUMINAMATH_CALUDE_cost_per_page_is_five_l395_39517

/-- Calculates the cost per page in cents -/
def cost_per_page (notebooks : ℕ) (pages_per_notebook : ℕ) (total_cost_dollars : ℕ) : ℚ :=
  (total_cost_dollars * 100) / (notebooks * pages_per_notebook)

/-- Proves that the cost per page is 5 cents given the problem conditions -/
theorem cost_per_page_is_five :
  cost_per_page 2 50 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_page_is_five_l395_39517


namespace NUMINAMATH_CALUDE_abs_eq_zero_iff_eq_seven_fifths_l395_39588

theorem abs_eq_zero_iff_eq_seven_fifths (x : ℝ) : |5*x - 7| = 0 ↔ x = 7/5 := by sorry

end NUMINAMATH_CALUDE_abs_eq_zero_iff_eq_seven_fifths_l395_39588


namespace NUMINAMATH_CALUDE_train_length_l395_39557

/-- Given a train that crosses a post in 15 seconds and a platform 100 m long in 25 seconds, its length is 150 meters. -/
theorem train_length (post_crossing_time platform_crossing_time platform_length : ℝ)
  (h1 : post_crossing_time = 15)
  (h2 : platform_crossing_time = 25)
  (h3 : platform_length = 100) :
  ∃ (train_length train_speed : ℝ),
    train_length = train_speed * post_crossing_time ∧
    train_length + platform_length = train_speed * platform_crossing_time ∧
    train_length = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l395_39557


namespace NUMINAMATH_CALUDE_linear_increasing_positive_slope_l395_39531

def f (k : ℝ) (x : ℝ) : ℝ := k * x - 100

theorem linear_increasing_positive_slope (k : ℝ) (h1 : k ≠ 0) :
  (∀ x y, x < y → f k x < f k y) → k > 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_increasing_positive_slope_l395_39531


namespace NUMINAMATH_CALUDE_temperature_drop_l395_39553

/-- Given an initial temperature and a temperature drop, calculate the final temperature. -/
def final_temperature (initial : ℤ) (drop : ℕ) : ℤ :=
  initial - drop

/-- Theorem: When the initial temperature is 3℃ and it drops by 5℃, the final temperature is -2℃. -/
theorem temperature_drop : final_temperature 3 5 = -2 := by
  sorry

end NUMINAMATH_CALUDE_temperature_drop_l395_39553


namespace NUMINAMATH_CALUDE_fixed_point_on_graph_l395_39596

theorem fixed_point_on_graph (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ a^(x + 2) - 3
  f (-2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_graph_l395_39596


namespace NUMINAMATH_CALUDE_prob_three_two_digit_l395_39561

/-- The number of dice being rolled -/
def num_dice : ℕ := 6

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The probability of rolling a two-digit number on a single die -/
def p_two_digit : ℚ := 11 / 20

/-- The probability of rolling a one-digit number on a single die -/
def p_one_digit : ℚ := 9 / 20

/-- The probability of exactly three dice showing a two-digit number when rolling 6 20-sided dice -/
theorem prob_three_two_digit : 
  (num_dice.choose 3 : ℚ) * p_two_digit ^ 3 * p_one_digit ^ 3 = 973971 / 3200000 :=
sorry

end NUMINAMATH_CALUDE_prob_three_two_digit_l395_39561


namespace NUMINAMATH_CALUDE_two_squares_same_plus_signs_l395_39589

/-- Represents a cell in the 8x8 table -/
structure Cell :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the 8x8 table with plus signs -/
def Table := Cell → Bool

/-- Represents a 4x4 square within the 8x8 table -/
structure Square :=
  (top_left_row : Fin 5)
  (top_left_col : Fin 5)

/-- Counts the number of plus signs in a given 4x4 square -/
def count_plus_signs (t : Table) (s : Square) : Nat :=
  sorry

theorem two_squares_same_plus_signs (t : Table) :
  ∃ s1 s2 : Square, s1 ≠ s2 ∧ count_plus_signs t s1 = count_plus_signs t s2 :=
sorry

end NUMINAMATH_CALUDE_two_squares_same_plus_signs_l395_39589


namespace NUMINAMATH_CALUDE_function_inequality_l395_39500

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≥ f y

-- State the theorem
theorem function_inequality :
  (∀ x, -8 < x → x < 8 → f x ≠ 0) →  -- f is defined on (-8, 8)
  is_even f →
  is_monotonic_on f 0 8 →
  f (-3) < f 2 →
  f 5 < f (-3) ∧ f (-3) < f (-1) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l395_39500


namespace NUMINAMATH_CALUDE_hundredths_place_of_five_eighths_l395_39539

theorem hundredths_place_of_five_eighths : ∃ (n : ℕ), (5 : ℚ) / 8 = (n * 100 + 20 : ℚ) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_hundredths_place_of_five_eighths_l395_39539


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l395_39568

open Set

-- Define the universal set I as the set of real numbers
def I : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {x | x < 1}

-- Theorem statement
theorem complement_M_intersect_N :
  (I \ M) ∩ N = {x : ℝ | x < -2} := by
  sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l395_39568


namespace NUMINAMATH_CALUDE_fuel_distance_theorem_l395_39545

/-- Represents the relationship between remaining fuel and distance traveled for a car -/
def fuel_distance_relation (initial_fuel : ℝ) (consumption_rate : ℝ) (x : ℝ) : ℝ :=
  initial_fuel - consumption_rate * x

/-- Theorem stating the relationship between remaining fuel and distance traveled -/
theorem fuel_distance_theorem (x : ℝ) :
  fuel_distance_relation 60 0.12 x = 60 - 0.12 * x := by
  sorry

end NUMINAMATH_CALUDE_fuel_distance_theorem_l395_39545


namespace NUMINAMATH_CALUDE_wendis_chickens_l395_39534

theorem wendis_chickens (initial : ℕ) 
  (h1 : 2 * initial - 1 + 6 = 13) : initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_wendis_chickens_l395_39534


namespace NUMINAMATH_CALUDE_vertex_of_our_parabola_l395_39594

/-- A parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Our specific parabola -/
def our_parabola : Parabola :=
  ⟨λ x => 2 * (x - 1)^2 + 5⟩

theorem vertex_of_our_parabola :
  vertex our_parabola = (1, 5) := by sorry

end NUMINAMATH_CALUDE_vertex_of_our_parabola_l395_39594


namespace NUMINAMATH_CALUDE_unique_solution_l395_39520

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define our equation
def equation (x : ℝ) : Prop := x ^ (floor x) = 9 / 2

-- State the theorem
theorem unique_solution : 
  ∃! x : ℝ, equation x ∧ x = (3 * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l395_39520


namespace NUMINAMATH_CALUDE_probability_at_least_one_l395_39578

theorem probability_at_least_one (A B : ℝ) (hA : A = 0.6) (hB : B = 0.7) 
  (h_independent : True) : 1 - (1 - A) * (1 - B) = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_l395_39578


namespace NUMINAMATH_CALUDE_first_number_equation_l395_39579

theorem first_number_equation (x : ℝ) : (x + 32 + 53) / 3 = (21 + 47 + 22) / 3 + 3 ↔ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_first_number_equation_l395_39579


namespace NUMINAMATH_CALUDE_pauline_bell_peppers_l395_39503

/-- The number of bell peppers Pauline bought -/
def num_bell_peppers : ℕ := 4

/-- The cost of taco shells in dollars -/
def taco_shells_cost : ℚ := 5

/-- The cost of each bell pepper in dollars -/
def bell_pepper_cost : ℚ := 3/2

/-- The cost of meat per pound in dollars -/
def meat_cost_per_pound : ℚ := 3

/-- The amount of meat Pauline bought in pounds -/
def meat_amount : ℚ := 2

/-- The total amount Pauline spent in dollars -/
def total_spent : ℚ := 17

theorem pauline_bell_peppers :
  num_bell_peppers = (total_spent - (taco_shells_cost + meat_cost_per_pound * meat_amount)) / bell_pepper_cost := by
  sorry

end NUMINAMATH_CALUDE_pauline_bell_peppers_l395_39503


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l395_39538

theorem fixed_point_parabola (k : ℝ) : 9 = 9 * (-1)^2 + k * (-1) - 5 * k := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l395_39538


namespace NUMINAMATH_CALUDE_sqrt_equation_l395_39529

theorem sqrt_equation (x y : ℝ) (h : Real.sqrt (x - 2) + (y - 3)^2 = 0) : 
  Real.sqrt (2*x + 3*y + 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l395_39529


namespace NUMINAMATH_CALUDE_expression_equality_l395_39536

theorem expression_equality : 
  Real.sqrt 12 + 2⁻¹ + Real.cos (60 * π / 180) - 3 * Real.tan (30 * π / 180) = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l395_39536


namespace NUMINAMATH_CALUDE_tan_two_pi_fifth_plus_theta_l395_39519

theorem tan_two_pi_fifth_plus_theta (θ : ℝ) 
  (h : Real.sin ((12 / 5) * Real.pi + θ) + 2 * Real.sin ((11 / 10) * Real.pi - θ) = 0) : 
  Real.tan ((2 / 5) * Real.pi + θ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_pi_fifth_plus_theta_l395_39519


namespace NUMINAMATH_CALUDE_intersection_y_coord_is_constant_l395_39560

/-- A point on the parabola y = x^2 -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = x^2

/-- The slope of the tangent line at a point on the parabola y = x^2 -/
def tangent_slope (p : ParabolaPoint) : ℝ := 2 * p.x

/-- Two points on the parabola y = x^2 with perpendicular tangents -/
structure PerpendicularTangentPoints where
  A : ParabolaPoint
  B : ParabolaPoint
  perpendicular : tangent_slope A * tangent_slope B = -1

/-- The y-coordinate of the intersection point of tangent lines -/
def intersection_y_coord (pts : PerpendicularTangentPoints) : ℝ :=
  pts.A.x * pts.B.x

theorem intersection_y_coord_is_constant (pts : PerpendicularTangentPoints) :
  intersection_y_coord pts = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_coord_is_constant_l395_39560


namespace NUMINAMATH_CALUDE_opposite_of_2023_l395_39566

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 → x = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l395_39566


namespace NUMINAMATH_CALUDE_all_normal_all_false_l395_39582

-- Define the possible types of people
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define the four people
structure Person :=
  (name : String)
  (type : PersonType)

-- Define the statements made
def statement1 (mr_b : Person) : Prop := mr_b.type = PersonType.Knight
def statement2 (mr_a : Person) (mr_b : Person) : Prop := 
  mr_a.type = PersonType.Knight ∧ mr_b.type = PersonType.Knight
def statement3 (mr_b : Person) : Prop := mr_b.type = PersonType.Knight

-- Define the problem setup
def problem_setup (mr_a mrs_a mr_b mrs_b : Person) : Prop :=
  mr_a.name = "Mr. A" ∧
  mrs_a.name = "Mrs. A" ∧
  mr_b.name = "Mr. B" ∧
  mrs_b.name = "Mrs. B"

-- Theorem statement
theorem all_normal_all_false 
  (mr_a mrs_a mr_b mrs_b : Person) 
  (h_setup : problem_setup mr_a mrs_a mr_b mrs_b) :
  (mr_a.type = PersonType.Normal ∧
   mrs_a.type = PersonType.Normal ∧
   mr_b.type = PersonType.Normal ∧
   mrs_b.type = PersonType.Normal) ∧
  (¬statement1 mr_b ∧
   ¬statement2 mr_a mr_b ∧
   ¬statement3 mr_b) :=
by sorry


end NUMINAMATH_CALUDE_all_normal_all_false_l395_39582


namespace NUMINAMATH_CALUDE_school_principal_election_l395_39586

/-- Given that Emma received 45 votes in a school principal election,
    and these votes represent 3/7 of the total votes,
    prove that the total number of votes cast is 105. -/
theorem school_principal_election (emma_votes : ℕ) (total_votes : ℕ)
    (h1 : emma_votes = 45)
    (h2 : emma_votes = 3 * total_votes / 7) :
    total_votes = 105 := by
  sorry

end NUMINAMATH_CALUDE_school_principal_election_l395_39586


namespace NUMINAMATH_CALUDE_angle_C_is_120_degrees_max_area_is_sqrt_3_l395_39571

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem angle_C_is_120_degrees (t : Triangle) 
  (h : t.a * (Real.cos t.C)^2 + 2 * t.c * Real.cos t.A * Real.cos t.C + t.a + t.b = 0) :
  t.C = 2 * π / 3 := by sorry

theorem max_area_is_sqrt_3 (t : Triangle) (h : t.b = 4 * Real.sin t.B) :
  (∀ u : Triangle, u.b = 4 * Real.sin u.B → t.a * t.b * Real.sin t.C / 2 ≥ u.a * u.b * Real.sin u.C / 2) ∧
  t.a * t.b * Real.sin t.C / 2 = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_C_is_120_degrees_max_area_is_sqrt_3_l395_39571


namespace NUMINAMATH_CALUDE_simplify_like_terms_l395_39576

theorem simplify_like_terms (y : ℝ) : 5 * y + 7 * y + 8 * y = 20 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_like_terms_l395_39576


namespace NUMINAMATH_CALUDE_subjective_not_set_l395_39564

-- Define what it means for a collection to have objective membership criteria
def has_objective_criteria (C : Type → Prop) : Prop :=
  ∀ (x : Type), (C x ∨ ¬C x) ∧ (∃ (f : Type → Bool), ∀ y, C y ↔ f y = true)

-- Define a set as a collection with objective membership criteria
def is_set (S : Type → Prop) : Prop := has_objective_criteria S

-- Define a collection with subjective criteria (e.g., "good friends")
def subjective_collection (x : Type) : Prop := sorry

-- Theorem: A collection with subjective criteria cannot form a set
theorem subjective_not_set : ¬(is_set subjective_collection) :=
sorry

end NUMINAMATH_CALUDE_subjective_not_set_l395_39564


namespace NUMINAMATH_CALUDE_triangle_third_altitude_l395_39535

theorem triangle_third_altitude (h₁ h₂ h₃ : ℝ) :
  h₁ = 8 → h₂ = 12 → h₃ > 0 →
  (1 / h₁ + 1 / h₂ > 1 / h₃) →
  h₃ > 4.8 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_altitude_l395_39535


namespace NUMINAMATH_CALUDE_four_tuple_solution_l395_39512

theorem four_tuple_solution (x y z w : ℝ) 
  (h1 : x^2 + y^2 + z^2 + w^2 = 4)
  (h2 : 1/x^2 + 1/y^2 + 1/z^2 + 1/w^2 = 5 - 1/(x*y*z*w)^2) :
  (x = 1 ∨ x = -1) ∧ 
  (y = 1 ∨ y = -1) ∧ 
  (z = 1 ∨ z = -1) ∧ 
  (w = 1 ∨ w = -1) ∧
  (x*y*z*w = 1 ∨ x*y*z*w = -1) :=
by sorry

end NUMINAMATH_CALUDE_four_tuple_solution_l395_39512


namespace NUMINAMATH_CALUDE_music_class_students_l395_39583

theorem music_class_students :
  ∃! n : ℕ, n < 50 ∧ n % 8 = 5 ∧ n % 6 = 3 ∧ n = 45 := by
sorry

end NUMINAMATH_CALUDE_music_class_students_l395_39583


namespace NUMINAMATH_CALUDE_solution_system_equations_l395_39533

theorem solution_system_equations :
  let x : ℝ := -1
  let y : ℝ := 2
  ((x^2 + y) * Real.sqrt (y - 2*x) - 4 = 2*x^2 + 2*x + y) ∧
  (x^3 - x^2 - y + 6 = 4 * Real.sqrt (x + 1) + 2 * Real.sqrt (y - 1)) := by
  sorry

end NUMINAMATH_CALUDE_solution_system_equations_l395_39533


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_99_l395_39505

theorem last_three_digits_of_7_to_99 : 7^99 ≡ 573 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_99_l395_39505


namespace NUMINAMATH_CALUDE_victor_percentage_proof_l395_39516

def max_marks : ℝ := 450
def victor_marks : ℝ := 405

theorem victor_percentage_proof :
  (victor_marks / max_marks) * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_victor_percentage_proof_l395_39516


namespace NUMINAMATH_CALUDE_one_tails_after_flips_l395_39543

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a circular arrangement of coins -/
def CoinCircle (n : ℕ) := Fin (2*n+1) → CoinState

/-- The initial state of the coin circle where all coins show heads -/
def initialState (n : ℕ) : CoinCircle n :=
  λ _ => CoinState.Heads

/-- The position of the k-th flip in the circle -/
def flipPosition (n k : ℕ) : Fin (2*n+1) :=
  ⟨k * (k + 1) / 2, sorry⟩

/-- Applies a single flip to a coin state -/
def flipCoin : CoinState → CoinState
| CoinState.Heads => CoinState.Tails
| CoinState.Tails => CoinState.Heads

/-- Applies the flipping process to the coin circle -/
def applyFlips (n : ℕ) (state : CoinCircle n) : CoinCircle n :=
  sorry

/-- Counts the number of tails in the final state -/
def countTails (n : ℕ) (state : CoinCircle n) : ℕ :=
  sorry

/-- The main theorem stating that exactly one coin shows tails after the process -/
theorem one_tails_after_flips (n : ℕ) :
  countTails n (applyFlips n (initialState n)) = 1 :=
sorry

end NUMINAMATH_CALUDE_one_tails_after_flips_l395_39543


namespace NUMINAMATH_CALUDE_current_age_proof_l395_39544

theorem current_age_proof (my_age : ℕ) (son_age : ℕ) : 
  (my_age - 9 = 5 * (son_age - 9)) →
  (my_age = 3 * son_age) →
  my_age = 54 := by
  sorry

end NUMINAMATH_CALUDE_current_age_proof_l395_39544


namespace NUMINAMATH_CALUDE_expensive_candy_price_l395_39593

/-- Given a mixture of two types of candy, prove the price of the more expensive candy. -/
theorem expensive_candy_price
  (total_weight : ℝ)
  (mixture_price : ℝ)
  (cheap_price : ℝ)
  (cheap_weight : ℝ)
  (h1 : total_weight = 80)
  (h2 : mixture_price = 2.20)
  (h3 : cheap_price = 2)
  (h4 : cheap_weight = 64) :
  ∃ (expensive_price : ℝ), expensive_price = 3 := by
sorry

end NUMINAMATH_CALUDE_expensive_candy_price_l395_39593


namespace NUMINAMATH_CALUDE_triangle_inequality_l395_39591

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 
   (b + c - a) / a + (c + a - b) / b + (a + b - c) / c) ∧
  ((b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l395_39591


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_l395_39597

theorem right_triangle_arithmetic_progression (a b c : ℝ) : 
  -- The triangle is right-angled
  a^2 + b^2 = c^2 →
  -- The lengths form an arithmetic progression
  b - a = c - b →
  -- The common difference is 1
  b - a = 1 →
  -- The hypotenuse is 5
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_l395_39597


namespace NUMINAMATH_CALUDE_circle_radius_from_arc_length_l395_39574

/-- Given a circle where the arc length corresponding to a central angle of 135° is 3π,
    prove that the radius of the circle is 4. -/
theorem circle_radius_from_arc_length :
  ∀ r : ℝ, (135 / 180 * π * r = 3 * π) → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_arc_length_l395_39574


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l395_39556

def A : Set Nat := {1,2,3,4,5}
def B : Set Nat := {2,4,6,8,10}

theorem union_of_A_and_B : A ∪ B = {1,2,3,4,5,6,8,10} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l395_39556


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_l395_39570

def i : ℂ := Complex.I

theorem pure_imaginary_complex (a : ℝ) : 
  (∃ (b : ℝ), (2 - i) * (a - i) = b * i ∧ b ≠ 0) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_l395_39570


namespace NUMINAMATH_CALUDE_salary_january_l395_39572

def employee_salary (jan feb mar apr may jun : ℕ) : Prop :=
  -- Average salary for Jan, Feb, Mar, Apr, May is Rs. 9,000
  (jan + feb + mar + apr + may) / 5 = 9000 ∧
  -- Average salary for Feb, Mar, Apr, May, Jun is Rs. 10,000
  (feb + mar + apr + may + jun) / 5 = 10000 ∧
  -- Employee receives a bonus of Rs. 1,500 in March
  ∃ (base_mar : ℕ), mar = base_mar + 1500 ∧
  -- Employee receives a deduction of Rs. 1,000 in June
  ∃ (base_jun : ℕ), jun = base_jun - 1000 ∧
  -- Salary for May is Rs. 7,500
  may = 7500 ∧
  -- No pay increase or decrease in the given time frame
  ∃ (base : ℕ), feb = base ∧ apr = base ∧ base_mar = base ∧ base_jun = base

theorem salary_january :
  ∀ (jan feb mar apr may jun : ℕ),
  employee_salary jan feb mar apr may jun →
  jan = 4500 :=
by sorry

end NUMINAMATH_CALUDE_salary_january_l395_39572


namespace NUMINAMATH_CALUDE_original_group_size_l395_39537

/-- Represents the work capacity of a group of men --/
def work_capacity (men : ℕ) (days : ℕ) : ℕ := men * days

theorem original_group_size
  (initial_days : ℕ)
  (absent_men : ℕ)
  (final_days : ℕ)
  (h1 : initial_days = 20)
  (h2 : absent_men = 10)
  (h3 : final_days = 40)
  : ∃ (original_size : ℕ),
    work_capacity original_size initial_days =
    work_capacity (original_size - absent_men) final_days ∧
    original_size = 20 :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_l395_39537


namespace NUMINAMATH_CALUDE_last_digits_divisible_by_three_l395_39581

theorem last_digits_divisible_by_three :
  ∃ (S : Finset Nat), (∀ n ∈ S, n < 10) ∧ (Finset.card S = 10) ∧
  (∀ d ∈ S, ∃ (m : Nat), m % 3 = 0 ∧ m % 10 = d) :=
sorry

end NUMINAMATH_CALUDE_last_digits_divisible_by_three_l395_39581


namespace NUMINAMATH_CALUDE_completing_square_l395_39580

theorem completing_square (x : ℝ) : x^2 - 2*x - 2 = 0 ↔ (x - 1)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l395_39580


namespace NUMINAMATH_CALUDE_total_milk_production_l395_39507

/-- 
Given two groups of cows with their respective milk production rates,
this theorem proves the total milk production for both groups over a specified period.
-/
theorem total_milk_production 
  (a b c x y z w : ℝ) 
  (ha : a > 0) 
  (hb : b ≥ 0) 
  (hc : c > 0) 
  (hx : x > 0) 
  (hy : y ≥ 0) 
  (hz : z > 0) 
  (hw : w ≥ 0) :
  let group_a_rate := b / c
  let group_b_rate := y / z
  (group_a_rate + group_b_rate) * w = b * w / c + y * w / z := by
  sorry

#check total_milk_production

end NUMINAMATH_CALUDE_total_milk_production_l395_39507


namespace NUMINAMATH_CALUDE_ratio_345_iff_arithmetic_sequence_l395_39558

/-- Represents a right-angled triangle with side lengths a, b, c where a < b < c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  a_lt_b : a < b
  b_lt_c : b < c
  right_angle : a^2 + b^2 = c^2

/-- The ratio of sides is 3:4:5 -/
def has_ratio_345 (t : RightTriangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.a = 3*k ∧ t.b = 4*k ∧ t.c = 5*k

/-- The sides form an arithmetic sequence -/
def is_arithmetic_sequence (t : RightTriangle) : Prop :=
  ∃ (d : ℝ), d > 0 ∧ t.b = t.a + d ∧ t.c = t.b + d

/-- The main theorem stating the equivalence of the two conditions -/
theorem ratio_345_iff_arithmetic_sequence (t : RightTriangle) :
  has_ratio_345 t ↔ is_arithmetic_sequence t :=
sorry

end NUMINAMATH_CALUDE_ratio_345_iff_arithmetic_sequence_l395_39558


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l395_39528

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 + b/a + a/b ≥ Real.sqrt 15 := by
  sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 + b/a + a/b = Real.sqrt 15 ↔ 
  a = (3/20)^(1/4) ∧ b = 1/(2*a) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l395_39528


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_for_not_p_necessary_not_sufficient_l395_39584

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0

def q (x : ℝ) : Prop := x^2 - 2*x - 8 ≤ 0 ∧ x^2 + 3*x - 10 > 0

-- Theorem for the first part of the problem
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) → x ∈ Set.Ioo 2 4 := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_not_p_necessary_not_sufficient :
  ∀ a : ℝ, (∀ x : ℝ, ¬(q x) → ¬(p x a)) ∧ (∃ x : ℝ, ¬(p x a) ∧ q x) → a ∈ Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_for_not_p_necessary_not_sufficient_l395_39584


namespace NUMINAMATH_CALUDE_smallest_m_is_12_l395_39595

-- Define the set T
def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

-- Define the property we want to prove
def has_nth_root_of_unity (n : ℕ) : Prop :=
  ∃ z ∈ T, z^n = 1

-- The main theorem
theorem smallest_m_is_12 :
  (∃ m : ℕ, m > 0 ∧ ∀ n ≥ m, has_nth_root_of_unity n) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ n ≥ m, has_nth_root_of_unity n) → m ≥ 12) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_12_l395_39595


namespace NUMINAMATH_CALUDE_common_area_rotated_squares_l395_39548

/-- The area of the region common to two squares with side length 2, 
    where one is rotated about a vertex by an angle θ such that cos θ = 3/5 -/
theorem common_area_rotated_squares (θ : Real) : 
  θ.cos = 3/5 → 
  (2 : Real) > 0 → 
  (4 * θ.cos * θ.sin : Real) = 48/25 := by
  sorry

end NUMINAMATH_CALUDE_common_area_rotated_squares_l395_39548


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l395_39518

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l395_39518


namespace NUMINAMATH_CALUDE_complex_argument_one_minus_i_sqrt_three_l395_39549

/-- The argument of the complex number 1 - i√3 is 5π/3 -/
theorem complex_argument_one_minus_i_sqrt_three (z : ℂ) : 
  z = 1 - Complex.I * Real.sqrt 3 → Complex.arg z = 5 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_argument_one_minus_i_sqrt_three_l395_39549


namespace NUMINAMATH_CALUDE_base7_addition_l395_39563

-- Define a function to convert base 7 numbers to natural numbers
def base7ToNat (a b c : Nat) : Nat :=
  a * 7^2 + b * 7 + c

-- Define the two numbers in base 7
def num1 : Nat := base7ToNat 0 2 5
def num2 : Nat := base7ToNat 2 4 6

-- Define the result in base 7
def result : Nat := base7ToNat 3 1 3

-- Theorem statement
theorem base7_addition :
  num1 + num2 = result := by
  sorry

end NUMINAMATH_CALUDE_base7_addition_l395_39563


namespace NUMINAMATH_CALUDE_kylie_coins_left_l395_39509

/-- The number of coins Kylie has left after collecting and giving away some coins -/
def coins_left (piggy_bank : ℕ) (from_brother : ℕ) (from_father : ℕ) (given_away : ℕ) : ℕ :=
  piggy_bank + from_brother + from_father - given_away

/-- Theorem stating that Kylie has 15 coins left given the problem conditions -/
theorem kylie_coins_left : coins_left 15 13 8 21 = 15 := by
  sorry

end NUMINAMATH_CALUDE_kylie_coins_left_l395_39509


namespace NUMINAMATH_CALUDE_base_number_proof_l395_39569

theorem base_number_proof (x : ℝ) : 16^8 = x^16 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l395_39569


namespace NUMINAMATH_CALUDE_constant_expression_l395_39573

theorem constant_expression (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 10)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 20) :
  5 * (a^3 + b^3 + c^3 + d^3 + e^3) - (a^4 + b^4 + c^4 + d^4 + e^4) = 120 := by
  sorry

end NUMINAMATH_CALUDE_constant_expression_l395_39573


namespace NUMINAMATH_CALUDE_sixth_number_is_52_when_i_7_l395_39515

/-- Represents the systematic sampling function for a population of 100 individuals -/
def systematicSample (i : Nat) (k : Nat) : Nat :=
  let drawnNumber := i + k
  if drawnNumber ≥ 10 then drawnNumber - 10 else drawnNumber

/-- Theorem stating that the 6th number drawn is 52 when i=7 -/
theorem sixth_number_is_52_when_i_7 :
  ∀ (populationSize : Nat) (segmentCount : Nat) (sampleSize : Nat) (i : Nat),
    populationSize = 100 →
    segmentCount = 10 →
    sampleSize = 10 →
    i = 7 →
    systematicSample i 5 = 52 := by
  sorry

end NUMINAMATH_CALUDE_sixth_number_is_52_when_i_7_l395_39515


namespace NUMINAMATH_CALUDE_dragons_games_count_l395_39555

theorem dragons_games_count :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (2 * initial_games / 5) →
    ∀ (final_games : ℕ),
      final_games = initial_games + 8 →
      (initial_wins + 8) = (3 * final_games / 5) →
      final_games = 24 :=
by sorry

end NUMINAMATH_CALUDE_dragons_games_count_l395_39555


namespace NUMINAMATH_CALUDE_water_fountain_length_l395_39525

/-- Given the conditions for building water fountains, prove the length of the fountain built by 20 men in 7 days -/
theorem water_fountain_length 
  (men1 : ℕ) (days1 : ℕ) (men2 : ℕ) (days2 : ℕ) (length2 : ℝ)
  (h1 : men1 = 20)
  (h2 : days1 = 7)
  (h3 : men2 = 35)
  (h4 : days2 = 3)
  (h5 : length2 = 42)
  (h_prop : ∀ (m d : ℕ) (l : ℝ), (m * d : ℝ) / (men2 * days2 : ℝ) = l / length2) :
  let length1 := (men1 * days1 : ℝ) * length2 / (men2 * days2 : ℝ)
  length1 = 56 := by
  sorry

end NUMINAMATH_CALUDE_water_fountain_length_l395_39525


namespace NUMINAMATH_CALUDE_line_intersects_plane_l395_39530

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relationships between points, lines, and planes
variable (on_line : Point → Line → Prop)
variable (in_plane : Point → Plane → Prop)
variable (intersects : Line → Plane → Prop)

-- Theorem statement
theorem line_intersects_plane (l : Line) (α : Plane) :
  (∃ p q : Point, on_line p l ∧ on_line q l ∧ in_plane p α ∧ ¬in_plane q α) →
  intersects l α :=
sorry

end NUMINAMATH_CALUDE_line_intersects_plane_l395_39530
