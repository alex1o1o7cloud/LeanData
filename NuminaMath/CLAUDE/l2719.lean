import Mathlib

namespace painter_can_blacken_all_cells_l2719_271936

/-- Represents a cell on the board -/
structure Cell :=
  (x : Nat) (y : Nat)

/-- Represents the color of a cell -/
inductive Color
  | Black
  | White

/-- Represents the board -/
def Board := Cell → Color

/-- Represents the painter's position -/
structure PainterPosition :=
  (cell : Cell)

/-- Function to change the color of a cell -/
def changeColor (color : Color) : Color :=
  match color with
  | Color.Black => Color.White
  | Color.White => Color.Black

/-- Function to check if a cell is on the border of the board -/
def isBorderCell (cell : Cell) (rows : Nat) (cols : Nat) : Prop :=
  cell.x = 0 ∨ cell.x = rows - 1 ∨ cell.y = 0 ∨ cell.y = cols - 1

/-- The main theorem -/
theorem painter_can_blacken_all_cells :
  ∀ (initialBoard : Board) (startPos : PainterPosition),
    (∀ (cell : Cell), cell.x < 2012 ∧ cell.y < 2013) →  -- Board dimensions
    (startPos.cell.x = 0 ∨ startPos.cell.x = 2011) ∧ (startPos.cell.y = 0 ∨ startPos.cell.y = 2012) →  -- Start from corner
    (∀ (cell : Cell), (cell.x + cell.y) % 2 = 0 → initialBoard cell = Color.Black) →  -- Initial checkerboard pattern
    (∀ (cell : Cell), (cell.x + cell.y) % 2 = 1 → initialBoard cell = Color.White) →
    ∃ (finalBoard : Board) (endPos : PainterPosition),
      (∀ (cell : Cell), finalBoard cell = Color.Black) ∧  -- All cells are black
      isBorderCell endPos.cell 2012 2013 :=  -- End on border
by sorry

end painter_can_blacken_all_cells_l2719_271936


namespace roots_product_theorem_l2719_271904

-- Define the polynomial f(x) = x⁶ + x³ + 1
def f (x : ℂ) : ℂ := x^6 + x^3 + 1

-- Define the function g(x) = x² + 1
def g (x : ℂ) : ℂ := x^2 + 1

-- State the theorem
theorem roots_product_theorem : 
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ : ℂ), 
    (∀ x, f x = (x - x₁) * (x - x₂) * (x - x₃) * (x - x₄) * (x - x₅) * (x - x₆)) →
    g x₁ * g x₂ * g x₃ * g x₄ * g x₅ * g x₆ = 1 := by
  sorry

end roots_product_theorem_l2719_271904


namespace smallest_cube_root_with_fractional_part_smallest_cube_root_exists_smallest_cube_root_is_68922_l2719_271994

theorem smallest_cube_root_with_fractional_part (m : ℕ) : 
  (∃ (n : ℕ) (r : ℝ), 
    n > 0 ∧ 
    r > 0 ∧ 
    r < 1/5000 ∧ 
    m^(1/3 : ℝ) = n + r) →
  m ≥ 68922 :=
by sorry

theorem smallest_cube_root_exists : 
  ∃ (n : ℕ) (r : ℝ), 
    n > 0 ∧ 
    r > 0 ∧ 
    r < 1/5000 ∧ 
    68922^(1/3 : ℝ) = n + r :=
by sorry

theorem smallest_cube_root_is_68922 : 
  (∀ m : ℕ, 
    (∃ (n : ℕ) (r : ℝ), 
      n > 0 ∧ 
      r > 0 ∧ 
      r < 1/5000 ∧ 
      m^(1/3 : ℝ) = n + r) →
    m ≥ 68922) ∧
  (∃ (n : ℕ) (r : ℝ), 
    n > 0 ∧ 
    r > 0 ∧ 
    r < 1/5000 ∧ 
    68922^(1/3 : ℝ) = n + r) :=
by sorry

end smallest_cube_root_with_fractional_part_smallest_cube_root_exists_smallest_cube_root_is_68922_l2719_271994


namespace line_equation_slope_intercept_l2719_271924

/-- The equation of a line with slope -1 and y-intercept -1 is x + y + 1 = 0 -/
theorem line_equation_slope_intercept (x y : ℝ) : 
  (∀ x y, y = -x - 1) ↔ (∀ x y, x + y + 1 = 0) :=
sorry

end line_equation_slope_intercept_l2719_271924


namespace longestAltitudesSum_eq_17_l2719_271971

/-- A triangle with sides 5, 12, and 13 -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13

/-- The sum of the lengths of the two longest altitudes in the special triangle -/
def longestAltitudesSum (t : SpecialTriangle) : ℝ := sorry

/-- Theorem stating that the sum of the lengths of the two longest altitudes is 17 -/
theorem longestAltitudesSum_eq_17 (t : SpecialTriangle) : longestAltitudesSum t = 17 := by sorry

end longestAltitudesSum_eq_17_l2719_271971


namespace product_evaluation_l2719_271974

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end product_evaluation_l2719_271974


namespace true_discount_calculation_l2719_271953

/-- Given the present worth and banker's gain, calculate the true discount -/
theorem true_discount_calculation (present_worth banker_gain : ℕ) 
  (h1 : present_worth = 576) 
  (h2 : banker_gain = 16) : 
  present_worth + banker_gain = 592 := by
  sorry

#check true_discount_calculation

end true_discount_calculation_l2719_271953


namespace zacks_friends_l2719_271961

def zacks_marbles : ℕ := 65
def marbles_kept : ℕ := 5
def marbles_per_friend : ℕ := 20

theorem zacks_friends :
  (zacks_marbles - marbles_kept) / marbles_per_friend = 3 :=
by sorry

end zacks_friends_l2719_271961


namespace guitar_payment_plan_l2719_271929

theorem guitar_payment_plan (total_with_interest : ℝ) (num_months : ℕ) (interest_rate : ℝ) :
  total_with_interest = 1320 →
  num_months = 12 →
  interest_rate = 0.1 →
  ∃ (monthly_payment : ℝ),
    monthly_payment * num_months * (1 + interest_rate) = total_with_interest ∧
    monthly_payment = 100 := by
  sorry

end guitar_payment_plan_l2719_271929


namespace minimum_fourth_quarter_score_l2719_271906

def required_average : ℝ := 85
def num_quarters : ℕ := 4
def first_quarter_score : ℝ := 84
def second_quarter_score : ℝ := 82
def third_quarter_score : ℝ := 80

theorem minimum_fourth_quarter_score :
  let total_required := required_average * num_quarters
  let current_total := first_quarter_score + second_quarter_score + third_quarter_score
  let minimum_fourth_score := total_required - current_total
  minimum_fourth_score = 94 ∧
  (first_quarter_score + second_quarter_score + third_quarter_score + minimum_fourth_score) / num_quarters ≥ required_average :=
by sorry

end minimum_fourth_quarter_score_l2719_271906


namespace choose_two_correct_l2719_271949

/-- The number of ways to choose 2 different items from n distinct items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that choose_two gives the correct number of ways to choose 2 from n -/
theorem choose_two_correct (n : ℕ) : choose_two n = Nat.choose n 2 := by
  sorry

end choose_two_correct_l2719_271949


namespace meaningful_fraction_l2719_271925

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (x + 3) / (x - 4)) ↔ x ≠ 4 := by sorry

end meaningful_fraction_l2719_271925


namespace cube_root_equation_solution_l2719_271920

theorem cube_root_equation_solution :
  ∃! x : ℝ, (10 - 2*x)^(1/3 : ℝ) = -2 :=
by
  -- The proof would go here
  sorry

end cube_root_equation_solution_l2719_271920


namespace inequality_proof_l2719_271916

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end inequality_proof_l2719_271916


namespace sum_of_integers_l2719_271943

theorem sum_of_integers (m n : ℕ+) 
  (h1 : m^2 + n^2 = 3789)
  (h2 : Nat.gcd m.val n.val + Nat.lcm m.val n.val = 633) : 
  m + n = 87 := by
  sorry

end sum_of_integers_l2719_271943


namespace rectangle_area_proof_l2719_271927

theorem rectangle_area_proof (l w : ℝ) : 
  (l + 3.5) * (w - 1.5) = l * w ∧ 
  (l - 3.5) * (w + 2) = l * w → 
  l * w = 630 := by
sorry

end rectangle_area_proof_l2719_271927


namespace coffee_cost_per_ounce_l2719_271952

/-- The cost of coffee per ounce, given the household consumption and weekly spending -/
theorem coffee_cost_per_ounce 
  (people : ℕ)
  (cups_per_person : ℕ)
  (ounces_per_cup : ℚ)
  (weekly_spending : ℚ) :
  people = 4 →
  cups_per_person = 2 →
  ounces_per_cup = 1/2 →
  weekly_spending = 35 →
  (weekly_spending / (people * cups_per_person * 7 * ounces_per_cup) : ℚ) = 5/4 := by
  sorry

end coffee_cost_per_ounce_l2719_271952


namespace museum_visit_l2719_271908

theorem museum_visit (num_students : ℕ) (ticket_price : ℕ) :
  (∃ k : ℕ, num_students = 5 * k) →
  (num_students + 1) * (ticket_price / 2) = 1599 →
  ticket_price % 2 = 0 →
  num_students = 40 ∧ ticket_price = 78 :=
by
  sorry

end museum_visit_l2719_271908


namespace prob_no_consecutive_ones_l2719_271902

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def validSequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of binary sequences of length n -/
def totalSequences (n : ℕ) : ℕ := 2^n

/-- Probability of no consecutive 1s in a sequence of length n -/
def prob (n : ℕ) : ℚ := (validSequences n : ℚ) / (totalSequences n : ℚ)

theorem prob_no_consecutive_ones : prob 12 = 377 / 4096 := by sorry

end prob_no_consecutive_ones_l2719_271902


namespace largest_interesting_is_correct_l2719_271966

/-- An interesting number is a natural number where all digits, except for the first and last,
    are less than the arithmetic mean of their two neighboring digits. -/
def is_interesting (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    digits[i]! < (digits[i-1]! + digits[i+1]!) / 2

/-- The largest interesting number -/
def largest_interesting : ℕ := 96433469

theorem largest_interesting_is_correct :
  is_interesting largest_interesting ∧
  ∀ m : ℕ, is_interesting m → m ≤ largest_interesting :=
sorry

end largest_interesting_is_correct_l2719_271966


namespace tv_horizontal_length_l2719_271922

/-- Calculates the horizontal length of a TV given its aspect ratio and diagonal length -/
theorem tv_horizontal_length 
  (aspect_width : ℝ) 
  (aspect_height : ℝ) 
  (diagonal_length : ℝ) 
  (aspect_width_pos : 0 < aspect_width)
  (aspect_height_pos : 0 < aspect_height)
  (diagonal_length_pos : 0 < diagonal_length) :
  let horizontal_length := aspect_width * diagonal_length / Real.sqrt (aspect_width^2 + aspect_height^2)
  horizontal_length = 16 * diagonal_length / Real.sqrt 337 :=
by sorry

end tv_horizontal_length_l2719_271922


namespace rotation_and_inclination_l2719_271985

/-- Given a point A(2,1) rotated counterclockwise around the origin O by π/4 to point B,
    if the angle of inclination of line OB is α, then cos α = √10/10 -/
theorem rotation_and_inclination :
  let A : ℝ × ℝ := (2, 1)
  let rotation_angle : ℝ := π / 4
  let B : ℝ × ℝ := (
    A.1 * Real.cos rotation_angle - A.2 * Real.sin rotation_angle,
    A.1 * Real.sin rotation_angle + A.2 * Real.cos rotation_angle
  )
  let α : ℝ := Real.arctan (B.2 / B.1)
  Real.cos α = Real.sqrt 10 / 10 := by sorry

end rotation_and_inclination_l2719_271985


namespace work_solution_l2719_271988

def work_problem (a b : ℝ) : Prop :=
  b = 15 ∧
  (3 / a + 5 * (1 / a + 1 / b) = 1) →
  a = 12

theorem work_solution : ∃ a b : ℝ, work_problem a b := by
  sorry

end work_solution_l2719_271988


namespace fly_path_total_distance_l2719_271942

theorem fly_path_total_distance (radius : ℝ) (leg : ℝ) (h1 : radius = 75) (h2 : leg = 70) :
  let diameter : ℝ := 2 * radius
  let other_leg : ℝ := Real.sqrt (diameter^2 - leg^2)
  diameter + leg + other_leg = 352.6 := by
sorry

end fly_path_total_distance_l2719_271942


namespace octagon_circles_theorem_l2719_271905

theorem octagon_circles_theorem (r : ℝ) (a b : ℤ) : 
  (∃ (s : ℝ), s = 2 ∧ s = r * Real.sqrt (2 - Real.sqrt 2)) →
  r^2 = a + b * Real.sqrt 2 →
  (a : ℝ) + b = 6 := by
sorry

end octagon_circles_theorem_l2719_271905


namespace alloy_b_ratio_l2719_271990

/-- Represents the composition of an alloy -/
structure Alloy where
  total_weight : ℝ
  tin_weight : ℝ
  lead_weight : ℝ
  copper_weight : ℝ

/-- The ratio of two components in an alloy -/
def ratio (a b : ℝ) : ℝ × ℝ := (a, b)

theorem alloy_b_ratio (alloy_a alloy_b : Alloy) (mixed_alloy : Alloy) :
  alloy_a.total_weight = 120 →
  alloy_b.total_weight = 180 →
  ratio alloy_a.lead_weight alloy_a.tin_weight = (2, 3) →
  mixed_alloy.tin_weight = 139.5 →
  mixed_alloy.total_weight = alloy_a.total_weight + alloy_b.total_weight →
  mixed_alloy.tin_weight = alloy_a.tin_weight + alloy_b.tin_weight →
  ratio alloy_b.tin_weight alloy_b.copper_weight = (3, 5) := by
  sorry

end alloy_b_ratio_l2719_271990


namespace jam_cost_is_348_l2719_271910

/-- The cost of jam used for all sandwiches --/
def jam_cost (N B J : ℕ) : ℚ :=
  (N * J * 6 : ℕ) / 100

/-- The total cost of ingredients for all sandwiches --/
def total_cost (N B J : ℕ) : ℚ :=
  (N * (5 * B + 6 * J) : ℕ) / 100

theorem jam_cost_is_348 (N B J : ℕ) :
  N > 1 ∧ B > 0 ∧ J > 0 ∧ total_cost N B J = 348 / 100 → jam_cost N B J = 348 / 100 := by
  sorry

end jam_cost_is_348_l2719_271910


namespace cost_price_of_ball_l2719_271937

theorem cost_price_of_ball (selling_price : ℕ) (num_balls : ℕ) (loss_balls : ℕ) :
  selling_price = 720 ∧ num_balls = 17 ∧ loss_balls = 5 →
  ∃ (cost_price : ℕ), cost_price * num_balls - cost_price * loss_balls = selling_price ∧ cost_price = 60 :=
by sorry

end cost_price_of_ball_l2719_271937


namespace probability_of_valid_pair_l2719_271954

/-- Represents a ball with a color and a label -/
structure Ball where
  color : Bool  -- True for red, False for blue
  label : Nat

/-- The bag of balls -/
def bag : Finset Ball := sorry

/-- The condition for a pair of balls to meet our criteria -/
def validPair (b1 b2 : Ball) : Prop :=
  b1.color ≠ b2.color ∧ b1.label + b2.label ≥ 4

/-- The number of ways to choose 2 balls from the bag -/
def totalChoices : Nat := sorry

/-- The number of valid pairs of balls -/
def validChoices : Nat := sorry

theorem probability_of_valid_pair :
  (validChoices : ℚ) / totalChoices = 3 / 10 := by sorry

end probability_of_valid_pair_l2719_271954


namespace square_neq_four_implies_neq_two_l2719_271915

theorem square_neq_four_implies_neq_two (a : ℝ) :
  (a^2 ≠ 4 → a ≠ 2) ∧ ¬(∀ a : ℝ, a ≠ 2 → a^2 ≠ 4) :=
by sorry

end square_neq_four_implies_neq_two_l2719_271915


namespace alternating_arrangement_count_l2719_271919

/-- The number of ways to arrange n elements from a set of m elements. -/
def A (m n : ℕ) : ℕ := sorry

/-- The number of ways to arrange 4 boys and 4 girls in a row,
    such that no two girls are adjacent and no two boys are adjacent. -/
def alternating_arrangement : ℕ := sorry

/-- Theorem stating that the number of alternating arrangements
    of 4 boys and 4 girls is equal to 2A₄⁴A₄⁴. -/
theorem alternating_arrangement_count :
  alternating_arrangement = 2 * A 4 4 * A 4 4 := by sorry

end alternating_arrangement_count_l2719_271919


namespace fraction_multiplication_l2719_271983

theorem fraction_multiplication (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (6 * x * y) / (5 * z^2) * (10 * z^3) / (9 * x * y) = (4 * z) / 3 := by
  sorry

end fraction_multiplication_l2719_271983


namespace simplify_sqrt_sum_l2719_271901

theorem simplify_sqrt_sum : 
  Real.sqrt (8 + 6 * Real.sqrt 3) + Real.sqrt (8 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end simplify_sqrt_sum_l2719_271901


namespace local_min_condition_l2719_271911

/-- The function f(x) = (x-1)e^x - ax has a local minimum point less than 0 
    if and only if a is in the open interval (-1/e, 0) -/
theorem local_min_condition (a : ℝ) : 
  (∃ x₀ < 0, IsLocalMin (fun x => (x - 1) * Real.exp x - a * x) x₀) ↔ 
  -1 / Real.exp 1 < a ∧ a < 0 := by
sorry

end local_min_condition_l2719_271911


namespace milk_fraction_after_transfers_l2719_271909

/-- Represents the contents of a mug --/
structure MugContents where
  tea : ℚ
  milk : ℚ

/-- Performs the liquid transfer operations as described in the problem --/
def transfer_liquids (initial_mug1 initial_mug2 : MugContents) : MugContents × MugContents :=
  sorry

/-- Calculates the fraction of milk in a mug --/
def milk_fraction (mug : MugContents) : ℚ :=
  mug.milk / (mug.tea + mug.milk)

theorem milk_fraction_after_transfers :
  let initial_mug1 : MugContents := { tea := 6, milk := 0 }
  let initial_mug2 : MugContents := { tea := 0, milk := 6 }
  let (final_mug1, _) := transfer_liquids initial_mug1 initial_mug2
  milk_fraction final_mug1 = 1/4 := by sorry

end milk_fraction_after_transfers_l2719_271909


namespace blocks_left_l2719_271921

theorem blocks_left (initial_blocks : ℕ) (used_blocks : ℕ) : 
  initial_blocks = 78 → used_blocks = 19 → initial_blocks - used_blocks = 59 := by
sorry

end blocks_left_l2719_271921


namespace randys_trip_length_l2719_271976

theorem randys_trip_length :
  ∀ (x : ℚ),
  (x / 4 : ℚ) + 40 + 10 + (x / 6 : ℚ) = x →
  x = 600 / 7 := by
sorry

end randys_trip_length_l2719_271976


namespace total_money_after_redistribution_l2719_271903

-- Define the redistribution function
def redistribute (a j t : ℚ) : ℚ × ℚ × ℚ :=
  let (a1, j1, t1) := (a - (j + t), 2*j, 2*t)
  let (a2, j2, t2) := (2*a1, j1 - (a1 + t1), 2*t1)
  (2*a2, 2*j2, t2 - (a2 + j2))

-- Theorem statement
theorem total_money_after_redistribution :
  ∀ a j : ℚ,
  let (a_final, j_final, t_final) := redistribute a j 24
  t_final = 24 →
  a_final + j_final + t_final = 168 := by
  sorry

end total_money_after_redistribution_l2719_271903


namespace division_and_addition_l2719_271935

theorem division_and_addition : (-300) / (-75) + 10 = 14 := by
  sorry

end division_and_addition_l2719_271935


namespace birthday_crayons_l2719_271975

/-- The number of crayons Paul got for his birthday -/
def initial_crayons : ℕ := 1453

/-- The number of crayons Paul gave away -/
def crayons_given : ℕ := 563

/-- The number of crayons Paul lost -/
def crayons_lost : ℕ := 558

/-- The number of crayons Paul had left -/
def crayons_left : ℕ := 332

/-- Theorem stating that the initial number of crayons equals the sum of crayons given away, lost, and left -/
theorem birthday_crayons : initial_crayons = crayons_given + crayons_lost + crayons_left := by
  sorry

end birthday_crayons_l2719_271975


namespace isosceles_triangle_figure_triangle_count_l2719_271960

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Represents the figure described in the problem -/
structure IsoscelesTriangleFigure where
  base : ℝ
  apex : Point
  baseLeft : Point
  baseRight : Point
  midpointLeft : Point
  midpointRight : Point

/-- Returns the number of triangles in the figure -/
def countTriangles (figure : IsoscelesTriangleFigure) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem isosceles_triangle_figure_triangle_count 
  (figure : IsoscelesTriangleFigure) 
  (h1 : figure.base = 2)
  (h2 : figure.baseLeft.y = figure.baseRight.y)
  (h3 : (figure.baseRight.x - figure.baseLeft.x) = figure.base)
  (h4 : figure.midpointLeft.x = (figure.baseLeft.x + figure.apex.x) / 2)
  (h5 : figure.midpointLeft.y = (figure.baseLeft.y + figure.apex.y) / 2)
  (h6 : figure.midpointRight.x = (figure.baseRight.x + figure.apex.x) / 2)
  (h7 : figure.midpointRight.y = (figure.baseRight.y + figure.apex.y) / 2)
  (h8 : figure.midpointLeft.y = figure.midpointRight.y) :
  countTriangles figure = 5 :=
sorry

end isosceles_triangle_figure_triangle_count_l2719_271960


namespace diophantine_equation_solutions_l2719_271938

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ,
    x ≤ y →
    x^2 + y^2 = 3 * 2016^z + 77 →
    ((x = 4 ∧ y = 8 ∧ z = 0) ∨
     (x = 14 ∧ y = 77 ∧ z = 1) ∨
     (x = 35 ∧ y = 70 ∧ z = 1)) :=
by sorry

end diophantine_equation_solutions_l2719_271938


namespace rhombus_diagonals_perpendicular_converse_is_false_inverse_is_false_contrapositive_l2719_271933

-- Define a type for quadrilaterals
structure Quadrilateral where
  -- Add necessary fields

-- Define what it means for a quadrilateral to be a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  sorry

-- Define what it means for diagonals to be perpendicular
def diagonals_perpendicular (q : Quadrilateral) : Prop :=
  sorry

-- The original statement
theorem rhombus_diagonals_perpendicular :
  ∀ q : Quadrilateral, is_rhombus q → diagonals_perpendicular q :=
sorry

-- The converse (which is false)
theorem converse_is_false :
  ¬(∀ q : Quadrilateral, diagonals_perpendicular q → is_rhombus q) :=
sorry

-- The inverse (which is false)
theorem inverse_is_false :
  ¬(∀ q : Quadrilateral, ¬is_rhombus q → ¬diagonals_perpendicular q) :=
sorry

-- The contrapositive (which is true)
theorem contrapositive :
  ∀ q : Quadrilateral, ¬diagonals_perpendicular q → ¬is_rhombus q :=
sorry

end rhombus_diagonals_perpendicular_converse_is_false_inverse_is_false_contrapositive_l2719_271933


namespace triangle_side_length_l2719_271998

theorem triangle_side_length (a b c : ℝ) : 
  a = 1 → b = 3 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  c ∈ ({3, 4, 5, 6} : Set ℝ) →
  c = 3 := by
sorry

end triangle_side_length_l2719_271998


namespace power_division_simplification_quadratic_expression_simplification_l2719_271912

-- Problem 1
theorem power_division_simplification :
  10^7 / (10^3 / 10^2) = 10^6 := by sorry

-- Problem 2
theorem quadratic_expression_simplification (x : ℝ) :
  (x + 2)^2 - (x + 1) * (x - 1) = 4 * x + 5 := by sorry

end power_division_simplification_quadratic_expression_simplification_l2719_271912


namespace race_participants_l2719_271995

theorem race_participants (first_year : ℕ) (second_year : ℕ) : 
  first_year = 8 →
  second_year = 5 * first_year →
  first_year + second_year = 48 := by
  sorry

end race_participants_l2719_271995


namespace simplify_expression_l2719_271964

theorem simplify_expression (a : ℝ) : 
  (1/2) * (8 * a^2 + 4 * a) - 3 * (a - (1/3) * a^2) = 5 * a^2 - a := by
  sorry

end simplify_expression_l2719_271964


namespace dave_ice_cubes_l2719_271969

/-- Given that Dave started with 2 ice cubes and ended with 9 ice cubes in total,
    prove that he made 7 additional ice cubes. -/
theorem dave_ice_cubes (initial : Nat) (final : Nat) (h1 : initial = 2) (h2 : final = 9) :
  final - initial = 7 := by
  sorry

end dave_ice_cubes_l2719_271969


namespace function_and_triangle_properties_l2719_271962

theorem function_and_triangle_properties 
  (ω : ℝ) 
  (h_ω_pos : ω > 0)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = 2 * Real.sin (ω * x) * Real.cos (ω * x) + 1)
  (h_period : ∀ x, f (x + 4 * Real.pi) = f x)
  (A B C : ℝ)
  (a b c : ℝ)
  (h_triangle : 2 * b * Real.cos A = a * Real.cos C + c * Real.cos A)
  (h_positive : 0 < A ∧ A < Real.pi) :
  ω = 1/2 ∧ 
  Real.cos A = 1/2 ∧ 
  A = Real.pi/3 ∧ 
  f A = Real.sqrt 3 / 2 := by
sorry

end function_and_triangle_properties_l2719_271962


namespace bacon_suggestion_count_bacon_suggestion_proof_l2719_271945

theorem bacon_suggestion_count : ℕ → ℕ → ℕ → Prop :=
  fun mashed_potatoes_count difference bacon_count =>
    (mashed_potatoes_count = 457) →
    (mashed_potatoes_count = bacon_count + difference) →
    (difference = 63) →
    (bacon_count = 394)

-- The proof is omitted
theorem bacon_suggestion_proof : bacon_suggestion_count 457 63 394 := by
  sorry

end bacon_suggestion_count_bacon_suggestion_proof_l2719_271945


namespace defeated_candidate_percentage_approx_l2719_271934

/-- Represents an election result -/
structure ElectionResult where
  total_votes : ℕ
  invalid_votes : ℕ
  margin_of_defeat : ℕ

/-- Calculates the percentage of votes for the defeated candidate -/
def defeated_candidate_percentage (result : ElectionResult) : ℚ :=
  let valid_votes := result.total_votes - result.invalid_votes
  let defeated_votes := (valid_votes - result.margin_of_defeat) / 2
  (defeated_votes : ℚ) / (valid_votes : ℚ) * 100

/-- Theorem stating that the percentage of votes for the defeated candidate is approximately 45.03% -/
theorem defeated_candidate_percentage_approx (result : ElectionResult)
  (h1 : result.total_votes = 90830)
  (h2 : result.invalid_votes = 83)
  (h3 : result.margin_of_defeat = 9000) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
  |defeated_candidate_percentage result - 45.03| < ε :=
sorry

end defeated_candidate_percentage_approx_l2719_271934


namespace a_less_than_b_l2719_271950

theorem a_less_than_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) 
  (h : (1 - a) * b > 1/4) : a < b := by
  sorry

end a_less_than_b_l2719_271950


namespace x_equals_one_l2719_271957

theorem x_equals_one (x y : ℕ+) 
  (h : ∀ n : ℕ+, (n * y)^2 + 1 ∣ x^(Nat.totient n) - 1) : 
  x = 1 := by
  sorry

end x_equals_one_l2719_271957


namespace y_range_given_x_constraints_l2719_271992

theorem y_range_given_x_constraints (x y : ℝ) 
  (h1 : 2 ≤ |x - 5| ∧ |x - 5| ≤ 9) 
  (h2 : y = 3 * x + 2) : 
  y ∈ Set.Icc (-10) 11 ∪ Set.Icc 23 44 := by
  sorry

end y_range_given_x_constraints_l2719_271992


namespace divisor_count_equality_implies_even_l2719_271993

/-- The number of positive integer divisors of n -/
def s (n : ℕ+) : ℕ := sorry

/-- If there exist positive integers a, b, and k such that k = s(a) = s(b) = s(2a+3b), then k must be even -/
theorem divisor_count_equality_implies_even (a b k : ℕ+) :
  k = s a ∧ k = s b ∧ k = s (2 * a + 3 * b) → Even k := by sorry

end divisor_count_equality_implies_even_l2719_271993


namespace sin_315_degrees_l2719_271914

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_315_degrees_l2719_271914


namespace mat_weavers_problem_l2719_271987

/-- The number of mat-weavers in the first group -/
def first_group_weavers : ℕ := sorry

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of mat-weavers in the second group -/
def second_group_weavers : ℕ := 12

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 36

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 12

theorem mat_weavers_problem :
  (first_group_mats : ℚ) / first_group_days / first_group_weavers =
  (second_group_mats : ℚ) / second_group_days / second_group_weavers →
  first_group_weavers = 4 := by
  sorry

end mat_weavers_problem_l2719_271987


namespace notebook_cost_l2719_271973

theorem notebook_cost (total_students : ℕ) (total_cost : ℕ) 
  (h_total_students : total_students = 36)
  (h_total_cost : total_cost = 2376)
  (s : ℕ) (n : ℕ) (c : ℕ)
  (h_majority : s > total_students / 2)
  (h_same_number : ∀ i j, i ≠ j → i < s → j < s → n = n)
  (h_at_least_two : n ≥ 2)
  (h_cost_greater : c > n)
  (h_total_equation : s * c * n = total_cost) :
  c = 11 := by
sorry

end notebook_cost_l2719_271973


namespace money_left_after_purchase_l2719_271951

def initial_money : ℕ := 56
def notebooks_bought : ℕ := 7
def books_bought : ℕ := 2
def notebook_cost : ℕ := 4
def book_cost : ℕ := 7

theorem money_left_after_purchase : 
  initial_money - (notebooks_bought * notebook_cost + books_bought * book_cost) = 14 :=
by sorry

end money_left_after_purchase_l2719_271951


namespace min_sum_squares_l2719_271941

theorem min_sum_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) 
  (heq : a^2 - 2015*a = b^2 - 2015*b) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x ≠ y → x^2 - 2015*x = y^2 - 2015*y → 
    x^2 + y^2 ≥ m) ∧ m = (2015^2 / 2) := by
  sorry

end min_sum_squares_l2719_271941


namespace chapter_page_difference_l2719_271982

theorem chapter_page_difference (first_chapter_pages second_chapter_pages : ℕ) 
  (h1 : first_chapter_pages = 48) 
  (h2 : second_chapter_pages = 11) : 
  first_chapter_pages - second_chapter_pages = 37 := by
  sorry

end chapter_page_difference_l2719_271982


namespace unique_c_for_unique_quadratic_solution_l2719_271986

theorem unique_c_for_unique_quadratic_solution :
  ∃! (c : ℝ), c ≠ 0 ∧
  (∃! (b : ℝ), b > 0 ∧
    (∃! (x : ℝ), x^2 + (b^2 + 1/b^2) * x + c = 0)) :=
by
  -- The proof goes here
  sorry

end unique_c_for_unique_quadratic_solution_l2719_271986


namespace earliest_meeting_time_l2719_271958

theorem earliest_meeting_time (david_lap_time maria_lap_time leo_lap_time : ℕ) 
  (h1 : david_lap_time = 5)
  (h2 : maria_lap_time = 8)
  (h3 : leo_lap_time = 10) :
  Nat.lcm (Nat.lcm david_lap_time maria_lap_time) leo_lap_time = 40 := by
sorry

end earliest_meeting_time_l2719_271958


namespace cone_volume_maximization_l2719_271948

theorem cone_volume_maximization (x : Real) : 
  let r := 1 -- radius of the original circular plate
  let cone_base_radius := (2 * Real.pi - x) / (2 * Real.pi) * r
  let cone_height := Real.sqrt (r ^ 2 - cone_base_radius ^ 2)
  let cone_volume := (1 / 3) * Real.pi * cone_base_radius ^ 2 * cone_height
  (∀ y, cone_volume ≤ (let cone_base_radius := (2 * Real.pi - y) / (2 * Real.pi) * r
                       let cone_height := Real.sqrt (r ^ 2 - cone_base_radius ^ 2)
                       (1 / 3) * Real.pi * cone_base_radius ^ 2 * cone_height)) →
  x = (6 - 2 * Real.sqrt 6) / 3 * Real.pi :=
by sorry

end cone_volume_maximization_l2719_271948


namespace alicia_book_cost_l2719_271923

/-- The total cost of books given the number of each type and their individual costs -/
def total_cost (math_books art_books science_books : ℕ) (math_cost art_cost science_cost : ℕ) : ℕ :=
  math_books * math_cost + art_books * art_cost + science_books * science_cost

/-- Theorem stating that the total cost of Alicia's books is $30 -/
theorem alicia_book_cost : total_cost 2 3 6 3 2 3 = 30 := by
  sorry

end alicia_book_cost_l2719_271923


namespace problem_statement_l2719_271932

/-- The problem statement --/
theorem problem_statement (x₀ y₀ r : ℝ) : 
  -- P(x₀, y₀) lies on both curves
  y₀ = 2 * Real.log x₀ ∧ 
  (x₀ - 3)^2 + y₀^2 = r^2 ∧ 
  -- Tangent lines are identical
  (2 / x₀ = -x₀ / y₀) ∧ 
  (2 / x₀ = x₀ * (y₀ - 2) / (9 - 3*x₀ - r^2)) ∧
  -- Quadratic function passes through (0,0), P(x₀, y₀), and (3,0)
  ∃ (a b c : ℝ), ∀ x, 
    (a*x^2 + b*x + c = 0) ∧
    (a*x₀^2 + b*x₀ + c = y₀) ∧
    (9*a + 3*b + c = 0) →
  -- The maximum value of the quadratic function is 9/8
  ∃ (f : ℝ → ℝ), (∀ x, f x ≤ 9/8) ∧ (∃ x, f x = 9/8) := by
sorry

end problem_statement_l2719_271932


namespace statements_c_and_d_are_correct_l2719_271979

theorem statements_c_and_d_are_correct :
  (∀ a b c : ℝ, c^2 > 0 → a*c^2 > b*c^2 → a > b) ∧
  (∀ a b m : ℝ, a > b → b > 0 → m > 0 → (b+m)/(a+m) > b/a) :=
by sorry

end statements_c_and_d_are_correct_l2719_271979


namespace inverse_proportional_symmetry_axis_l2719_271999

theorem inverse_proportional_symmetry_axis (k : ℝ) (h1 : k ≠ 0) (h2 : k ≠ 1) :
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = k / x) ∧
  (∀ x ≠ 0, ∃ y, f y = f x ∧ (y + x) * (-k / |k|) = y - x) :=
sorry

end inverse_proportional_symmetry_axis_l2719_271999


namespace probability_three_odd_dice_l2719_271939

theorem probability_three_odd_dice (n : ℕ) (p : ℝ) : 
  n = 5 →                          -- number of dice
  p = 1 / 2 →                      -- probability of rolling an odd number on a single die
  (Nat.choose n 3 : ℝ) * p^3 * (1 - p)^(n - 3) = 5 / 16 :=
by sorry

end probability_three_odd_dice_l2719_271939


namespace pure_imaginary_condition_l2719_271931

theorem pure_imaginary_condition (m : ℝ) : 
  (((2 : ℂ) - m * Complex.I) / (1 + Complex.I)).re = 0 → m = 2 :=
by
  sorry

end pure_imaginary_condition_l2719_271931


namespace sand_tank_mass_l2719_271913

/-- Given a tank filled with sand, prove that the total mass when completely filled
    is (8p - 3q) / 5, where p is the mass when 3/4 filled and q is the mass when 1/3 filled. -/
theorem sand_tank_mass (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p > q) :
  ∃ (z w : ℝ), z > 0 ∧ w > 0 ∧
    z + 3/4 * w = p ∧
    z + 1/3 * w = q ∧
    z + w = (8*p - 3*q) / 5 :=
by sorry

end sand_tank_mass_l2719_271913


namespace nabla_equation_solution_l2719_271996

/-- The nabla operation defined for real numbers -/
def nabla (a b : ℝ) : ℝ := (a + 1) * (b - 2)

/-- Theorem: If 5 ∇ x = 30, then x = 7 -/
theorem nabla_equation_solution :
  ∀ x : ℝ, nabla 5 x = 30 → x = 7 := by
  sorry

end nabla_equation_solution_l2719_271996


namespace function_range_l2719_271972

/-- Given a^2 - a < 2 and a is a positive integer, 
    the range of f(x) = x + 2a/x is (-∞, -2√2] ∪ [2√2, +∞) -/
theorem function_range (a : ℕ+) (h : a^2 - a < 2) :
  Set.range (fun x : ℝ => x + 2*a/x) = 
    Set.Iic (-2 * Real.sqrt 2) ∪ Set.Ici (2 * Real.sqrt 2) := by
  sorry

end function_range_l2719_271972


namespace solve_equation_l2719_271955

theorem solve_equation : ∃ x : ℝ, 2.25 * x = 45 ∧ x = 20 := by
  sorry

end solve_equation_l2719_271955


namespace quadrilateral_reconstruction_l2719_271981

-- Define the quadrilateral and its points
variable (E F G H E' F' G' H' : ℝ × ℝ)

-- Define the conditions
variable (h1 : E' - F = E - F)
variable (h2 : F' - G = F - G)
variable (h3 : G' - H = G - H)
variable (h4 : H' - E = H - E)

-- Define the theorem
theorem quadrilateral_reconstruction :
  ∃ (x y z w : ℝ),
    E = x • E' + y • F' + z • G' + w • H' ∧
    x = 1/15 ∧ y = 2/15 ∧ z = 4/15 ∧ w = 8/15 :=
by sorry

end quadrilateral_reconstruction_l2719_271981


namespace absolute_difference_equation_l2719_271944

theorem absolute_difference_equation : 
  ∃! x : ℝ, |16 - x| - |x - 12| = 4 :=
by
  sorry

end absolute_difference_equation_l2719_271944


namespace problem_solution_l2719_271956

theorem problem_solution (x y : ℝ) (h1 : x - y = 1) (h2 : x^3 - y^3 = 2) :
  x^4 + y^4 = 23/9 ∧ x^5 - y^5 = 29/9 := by
  sorry

end problem_solution_l2719_271956


namespace allison_wins_prob_l2719_271984

/-- Represents a 6-sided cube with specified face values -/
structure Cube where
  faces : Fin 6 → ℕ

/-- Allison's cube with all faces showing 6 -/
def allison_cube : Cube :=
  { faces := fun _ => 6 }

/-- Brian's cube with faces numbered 1 to 6 -/
def brian_cube : Cube :=
  { faces := fun i => i.val + 1 }

/-- Noah's cube with three faces showing 3 and three faces showing 5 -/
def noah_cube : Cube :=
  { faces := fun i => if i.val < 3 then 3 else 5 }

/-- The probability of rolling a value less than n on a given cube -/
def prob_less_than (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (fun i => c.faces i < n) (Finset.univ : Finset (Fin 6))).card / 6

/-- The main theorem stating the probability of Allison's roll being greater than both Brian's and Noah's -/
theorem allison_wins_prob :
    prob_less_than brian_cube 6 * prob_less_than noah_cube 6 = 5 / 12 := by
  sorry

end allison_wins_prob_l2719_271984


namespace quadratic_passes_through_points_l2719_271968

/-- A quadratic function passing through the points (2,0), (0,4), and (-2,0) -/
def quadratic_function (x : ℝ) : ℝ := -x^2 + 4

/-- Theorem stating that the quadratic function passes through the given points -/
theorem quadratic_passes_through_points :
  (quadratic_function 2 = 0) ∧
  (quadratic_function 0 = 4) ∧
  (quadratic_function (-2) = 0) := by
  sorry

end quadratic_passes_through_points_l2719_271968


namespace lunch_average_price_proof_l2719_271917

theorem lunch_average_price_proof (total_price : ℝ) (num_people : ℕ) (gratuity_rate : ℝ) 
  (h1 : total_price = 207)
  (h2 : num_people = 15)
  (h3 : gratuity_rate = 0.15) :
  (total_price / (1 + gratuity_rate)) / num_people = 12 := by
  sorry

end lunch_average_price_proof_l2719_271917


namespace min_marked_elements_eq_666_l2719_271946

/-- The minimum number of marked elements in {1, ..., 2000} such that
    for every pair (k, 2k) where 1 ≤ k ≤ 1000, at least one of k or 2k is marked. -/
def min_marked_elements : ℕ :=
  let S := Finset.range 2000
  Finset.filter (fun n => ∃ k ∈ Finset.range 1000, n = k ∨ n = 2 * k) S |>.card

/-- The theorem stating that the minimum number of marked elements is 666. -/
theorem min_marked_elements_eq_666 : min_marked_elements = 666 := by
  sorry

end min_marked_elements_eq_666_l2719_271946


namespace apple_difference_l2719_271978

/-- An apple eating contest with six students -/
structure AppleContest where
  students : Nat
  max_apples : Nat
  min_apples : Nat

/-- The properties of the given apple eating contest -/
def given_contest : AppleContest :=
  { students := 6
  , max_apples := 6
  , min_apples := 1 }

/-- Theorem stating the difference between max and min apples eaten -/
theorem apple_difference (contest : AppleContest) (h1 : contest = given_contest) :
  contest.max_apples - contest.min_apples = 5 := by
  sorry

end apple_difference_l2719_271978


namespace joyce_initial_apples_l2719_271930

/-- The number of apples Joyce started with -/
def initial_apples : ℕ := sorry

/-- The number of apples Larry gave to Joyce -/
def apples_from_larry : ℚ := 52.0

/-- The total number of apples Joyce has after receiving apples from Larry -/
def total_apples : ℕ := 127

/-- Theorem stating that Joyce started with 75 apples -/
theorem joyce_initial_apples :
  initial_apples = 75 :=
by sorry

end joyce_initial_apples_l2719_271930


namespace distance_to_midpoint_l2719_271989

/-- The distance from Shinyoung's house to the midpoint of the path to school -/
theorem distance_to_midpoint (house_to_office village_to_school : ℕ) : 
  house_to_office = 1700 →
  village_to_school = 900 →
  (house_to_office + village_to_school) / 2 = 1300 := by
  sorry

end distance_to_midpoint_l2719_271989


namespace inverse_variation_problem_l2719_271997

theorem inverse_variation_problem (z x : ℝ) (h : ∃ k : ℝ, ∀ x z, z * x^2 = k) :
  (2 * 3^2 = z * 3^2) → (8 * x^2 = z * 3^2) → x = 3/2 := by
  sorry

end inverse_variation_problem_l2719_271997


namespace probability_yellow_ball_l2719_271926

/-- The probability of drawing a yellow ball from a bag containing white and yellow balls -/
theorem probability_yellow_ball (total_balls : ℕ) (white_balls yellow_balls : ℕ) 
  (h1 : total_balls = white_balls + yellow_balls)
  (h2 : total_balls > 0)
  (h3 : white_balls = 2)
  (h4 : yellow_balls = 3) :
  (yellow_balls : ℚ) / total_balls = 3 / 5 :=
by sorry

end probability_yellow_ball_l2719_271926


namespace base_5_of_156_l2719_271967

/-- Converts a natural number to its base 5 representation as a list of digits --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Theorem: The base 5 representation of 156 (base 10) is [1, 1, 1, 1] --/
theorem base_5_of_156 : toBase5 156 = [1, 1, 1, 1] := by
  sorry

end base_5_of_156_l2719_271967


namespace plane_equation_l2719_271959

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space represented by parametric equations -/
structure Line3D where
  x : ℝ → ℝ
  y : ℝ → ℝ
  z : ℝ → ℝ

/-- A plane in 3D space represented by Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

def point_on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

def line_on_plane (l : Line3D) (plane : Plane) : Prop :=
  ∀ t : ℝ, point_on_plane ⟨l.x t, l.y t, l.z t⟩ plane

def is_solution (plane : Plane) : Prop :=
  let p1 : Point3D := ⟨1, 4, -3⟩
  let p2 : Point3D := ⟨0, -3, 0⟩
  let l : Line3D := { x := λ t => 4 * t + 2, y := λ t => -t - 2, z := λ t => 5 * t + 1 }
  point_on_plane p1 plane ∧
  point_on_plane p2 plane ∧
  line_on_plane l plane ∧
  plane.A > 0 ∧
  Nat.gcd (Int.natAbs plane.A) (Nat.gcd (Int.natAbs plane.B) (Nat.gcd (Int.natAbs plane.C) (Int.natAbs plane.D))) = 1

theorem plane_equation : is_solution ⟨10, 9, -13, 27⟩ := by
  sorry

end plane_equation_l2719_271959


namespace roots_position_l2719_271907

theorem roots_position (a b : ℝ) :
  ∃ (x₁ x₂ : ℝ), (x₁ - a) * (x₁ - a - b) = 1 ∧
                  (x₂ - a) * (x₂ - a - b) = 1 ∧
                  x₁ < a ∧ a < x₂ := by
  sorry

end roots_position_l2719_271907


namespace anthony_transaction_percentage_l2719_271918

theorem anthony_transaction_percentage (mabel_transactions cal_transactions jade_transactions : ℕ)
  (anthony_transactions : ℕ) :
  mabel_transactions = 90 →
  cal_transactions = (2 : ℚ) / 3 * anthony_transactions →
  jade_transactions = cal_transactions + 14 →
  jade_transactions = 80 →
  (anthony_transactions - mabel_transactions : ℚ) / mabel_transactions * 100 = 10 := by
sorry

end anthony_transaction_percentage_l2719_271918


namespace worker_arrival_time_l2719_271900

/-- Proves that a worker walking at 4/5 of her normal speed arrives 10 minutes later -/
theorem worker_arrival_time (normal_speed : ℝ) (normal_time : ℝ) 
  (h1 : normal_time = 40)
  (h2 : normal_speed > 0) :
  let reduced_speed := (4/5 : ℝ) * normal_speed
  let new_time := normal_time * (normal_speed / reduced_speed)
  new_time - normal_time = 10 := by
sorry


end worker_arrival_time_l2719_271900


namespace red_balls_count_l2719_271947

/-- Represents a box containing white and red balls -/
structure BallBox where
  white_balls : ℕ
  red_balls : ℕ

/-- The probability of picking a red ball from the box -/
def red_probability (box : BallBox) : ℚ :=
  box.red_balls / (box.white_balls + box.red_balls)

/-- Theorem: If there are 12 white balls and the probability of picking a red ball is 1/4,
    then the number of red balls is 4 -/
theorem red_balls_count (box : BallBox) 
    (h1 : box.white_balls = 12)
    (h2 : red_probability box = 1/4) : 
    box.red_balls = 4 := by
  sorry

end red_balls_count_l2719_271947


namespace childrens_vehicle_wheels_l2719_271940

theorem childrens_vehicle_wheels 
  (adult_count : ℕ) 
  (child_count : ℕ) 
  (total_wheels : ℕ) 
  (bicycle_wheels : ℕ) :
  adult_count = 6 →
  child_count = 15 →
  total_wheels = 57 →
  bicycle_wheels = 2 →
  ∃ (child_vehicle_wheels : ℕ), 
    child_vehicle_wheels = 3 ∧
    total_wheels = adult_count * bicycle_wheels + child_count * child_vehicle_wheels :=
by sorry

end childrens_vehicle_wheels_l2719_271940


namespace store_revenue_comparison_l2719_271977

theorem store_revenue_comparison (december : ℝ) (november : ℝ) (january : ℝ)
  (h1 : november = (2/5) * december)
  (h2 : january = (1/5) * november) :
  december = (25/6) * ((november + january) / 2) := by
  sorry

end store_revenue_comparison_l2719_271977


namespace polynomial_equality_l2719_271970

-- Define the theorem
theorem polynomial_equality (n : ℕ) (f g : ℝ → ℝ) (x : Fin (n + 1) → ℝ) :
  (∀ (k : Fin (n + 1)), (deriv^[k] f) (x k) = (deriv^[k] g) (x k)) →
  (∀ (y : ℝ), f y = g y) :=
sorry

end polynomial_equality_l2719_271970


namespace grapefruit_orchards_count_l2719_271928

/-- Calculates the number of grapefruit orchards in a citrus grove. -/
def grapefruit_orchards (total : ℕ) (lemon : ℕ) : ℕ :=
  let orange := lemon / 2
  let remaining := total - (lemon + orange)
  remaining / 2

/-- Proves that the number of grapefruit orchards is 2 given the specified conditions. -/
theorem grapefruit_orchards_count :
  grapefruit_orchards 16 8 = 2 := by
  sorry

end grapefruit_orchards_count_l2719_271928


namespace cubic_root_function_l2719_271991

/-- Given a function y = kx^(1/3) where y = 5√2 when x = 64, 
    prove that y = 2.5√2 when x = 8 -/
theorem cubic_root_function (k : ℝ) : 
  (∃ y : ℝ, y = k * 64^(1/3) ∧ y = 5 * Real.sqrt 2) →
  k * 8^(1/3) = 2.5 * Real.sqrt 2 := by
  sorry


end cubic_root_function_l2719_271991


namespace maximize_product_l2719_271965

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 28) :
  x^5 * y^3 ≤ 17.5^5 * 10.5^3 ∧
  (x^5 * y^3 = 17.5^5 * 10.5^3 ↔ x = 17.5 ∧ y = 10.5) :=
by sorry

end maximize_product_l2719_271965


namespace train_speed_proof_l2719_271980

/-- Proves that a train crossing a 280-meter platform in 30 seconds and passing a stationary man in 16 seconds has a speed of 72 km/h -/
theorem train_speed_proof (platform_length : Real) (platform_crossing_time : Real) 
  (man_passing_time : Real) (speed_kmh : Real) : 
  platform_length = 280 ∧ 
  platform_crossing_time = 30 ∧ 
  man_passing_time = 16 ∧
  speed_kmh = (platform_length / (platform_crossing_time - man_passing_time)) * 3.6 →
  speed_kmh = 72 := by
sorry

end train_speed_proof_l2719_271980


namespace bus_speed_excluding_stoppages_l2719_271963

/-- Given a bus that stops for half an hour every hour and has an average speed of 6 km/hr including stoppages, 
    its speed excluding stoppages is 12 km/hr. -/
theorem bus_speed_excluding_stoppages 
  (stop_time : ℝ) 
  (avg_speed_with_stops : ℝ) 
  (h1 : stop_time = 0.5) -- 30 minutes = 0.5 hours
  (h2 : avg_speed_with_stops = 6) :
  avg_speed_with_stops / (1 - stop_time) = 12 := by
  sorry

#check bus_speed_excluding_stoppages

end bus_speed_excluding_stoppages_l2719_271963
