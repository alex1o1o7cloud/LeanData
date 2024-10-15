import Mathlib

namespace NUMINAMATH_CALUDE_chapter_page_difference_l75_7569

theorem chapter_page_difference (first_chapter_pages second_chapter_pages : ℕ) 
  (h1 : first_chapter_pages = 48) 
  (h2 : second_chapter_pages = 11) : 
  first_chapter_pages - second_chapter_pages = 37 := by
  sorry

end NUMINAMATH_CALUDE_chapter_page_difference_l75_7569


namespace NUMINAMATH_CALUDE_store_revenue_comparison_l75_7595

theorem store_revenue_comparison (december : ℝ) (november : ℝ) (january : ℝ)
  (h1 : november = (2/5) * december)
  (h2 : january = (1/5) * november) :
  december = (25/6) * ((november + january) / 2) := by
  sorry

end NUMINAMATH_CALUDE_store_revenue_comparison_l75_7595


namespace NUMINAMATH_CALUDE_rhombus_diagonals_perpendicular_converse_is_false_inverse_is_false_contrapositive_l75_7509

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

end NUMINAMATH_CALUDE_rhombus_diagonals_perpendicular_converse_is_false_inverse_is_false_contrapositive_l75_7509


namespace NUMINAMATH_CALUDE_min_sum_squares_l75_7589

theorem min_sum_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) 
  (heq : a^2 - 2015*a = b^2 - 2015*b) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x ≠ y → x^2 - 2015*x = y^2 - 2015*y → 
    x^2 + y^2 ≥ m) ∧ m = (2015^2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l75_7589


namespace NUMINAMATH_CALUDE_painter_can_blacken_all_cells_l75_7562

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

end NUMINAMATH_CALUDE_painter_can_blacken_all_cells_l75_7562


namespace NUMINAMATH_CALUDE_a_less_than_b_l75_7549

theorem a_less_than_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) 
  (h : (1 - a) * b > 1/4) : a < b := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_b_l75_7549


namespace NUMINAMATH_CALUDE_fly_path_total_distance_l75_7551

theorem fly_path_total_distance (radius : ℝ) (leg : ℝ) (h1 : radius = 75) (h2 : leg = 70) :
  let diameter : ℝ := 2 * radius
  let other_leg : ℝ := Real.sqrt (diameter^2 - leg^2)
  diameter + leg + other_leg = 352.6 := by
sorry

end NUMINAMATH_CALUDE_fly_path_total_distance_l75_7551


namespace NUMINAMATH_CALUDE_coffee_cost_per_ounce_l75_7591

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

end NUMINAMATH_CALUDE_coffee_cost_per_ounce_l75_7591


namespace NUMINAMATH_CALUDE_product_evaluation_l75_7575

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l75_7575


namespace NUMINAMATH_CALUDE_minimum_fourth_quarter_score_l75_7539

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

end NUMINAMATH_CALUDE_minimum_fourth_quarter_score_l75_7539


namespace NUMINAMATH_CALUDE_prob_no_consecutive_ones_l75_7578

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

end NUMINAMATH_CALUDE_prob_no_consecutive_ones_l75_7578


namespace NUMINAMATH_CALUDE_fraction_1800_1809_l75_7501

/-- The number of states that joined the union during 1800-1809. -/
def states_1800_1809 : ℕ := 4

/-- The total number of states in Walter's collection. -/
def total_states : ℕ := 30

/-- The fraction of states that joined during 1800-1809 out of the first 30 states. -/
theorem fraction_1800_1809 : (states_1800_1809 : ℚ) / total_states = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_1800_1809_l75_7501


namespace NUMINAMATH_CALUDE_mat_weavers_problem_l75_7533

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

end NUMINAMATH_CALUDE_mat_weavers_problem_l75_7533


namespace NUMINAMATH_CALUDE_lunch_average_price_proof_l75_7588

theorem lunch_average_price_proof (total_price : ℝ) (num_people : ℕ) (gratuity_rate : ℝ) 
  (h1 : total_price = 207)
  (h2 : num_people = 15)
  (h3 : gratuity_rate = 0.15) :
  (total_price / (1 + gratuity_rate)) / num_people = 12 := by
  sorry

end NUMINAMATH_CALUDE_lunch_average_price_proof_l75_7588


namespace NUMINAMATH_CALUDE_problem_solution_l75_7570

theorem problem_solution (x y : ℝ) (h1 : x - y = 1) (h2 : x^3 - y^3 = 2) :
  x^4 + y^4 = 23/9 ∧ x^5 - y^5 = 29/9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l75_7570


namespace NUMINAMATH_CALUDE_base_5_of_156_l75_7526

/-- Converts a natural number to its base 5 representation as a list of digits --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Theorem: The base 5 representation of 156 (base 10) is [1, 1, 1, 1] --/
theorem base_5_of_156 : toBase5 156 = [1, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_5_of_156_l75_7526


namespace NUMINAMATH_CALUDE_quadrilateral_reconstruction_l75_7568

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

end NUMINAMATH_CALUDE_quadrilateral_reconstruction_l75_7568


namespace NUMINAMATH_CALUDE_longestAltitudesSum_eq_17_l75_7520

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

end NUMINAMATH_CALUDE_longestAltitudesSum_eq_17_l75_7520


namespace NUMINAMATH_CALUDE_polynomial_equality_l75_7538

-- Define the theorem
theorem polynomial_equality (n : ℕ) (f g : ℝ → ℝ) (x : Fin (n + 1) → ℝ) :
  (∀ (k : Fin (n + 1)), (deriv^[k] f) (x k) = (deriv^[k] g) (x k)) →
  (∀ (y : ℝ), f y = g y) :=
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l75_7538


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l75_7577

theorem simplify_sqrt_sum : 
  Real.sqrt (8 + 6 * Real.sqrt 3) + Real.sqrt (8 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l75_7577


namespace NUMINAMATH_CALUDE_zacks_friends_l75_7582

def zacks_marbles : ℕ := 65
def marbles_kept : ℕ := 5
def marbles_per_friend : ℕ := 20

theorem zacks_friends :
  (zacks_marbles - marbles_kept) / marbles_per_friend = 3 :=
by sorry

end NUMINAMATH_CALUDE_zacks_friends_l75_7582


namespace NUMINAMATH_CALUDE_alloy_b_ratio_l75_7513

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

end NUMINAMATH_CALUDE_alloy_b_ratio_l75_7513


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l75_7579

theorem inverse_variation_problem (z x : ℝ) (h : ∃ k : ℝ, ∀ x z, z * x^2 = k) :
  (2 * 3^2 = z * 3^2) → (8 * x^2 = z * 3^2) → x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l75_7579


namespace NUMINAMATH_CALUDE_birthday_crayons_l75_7527

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

end NUMINAMATH_CALUDE_birthday_crayons_l75_7527


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l75_7561

theorem rectangle_area_proof (l w : ℝ) : 
  (l + 3.5) * (w - 1.5) = l * w ∧ 
  (l - 3.5) * (w + 2) = l * w → 
  l * w = 630 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l75_7561


namespace NUMINAMATH_CALUDE_divisor_count_equality_implies_even_l75_7514

/-- The number of positive integer divisors of n -/
def s (n : ℕ+) : ℕ := sorry

/-- If there exist positive integers a, b, and k such that k = s(a) = s(b) = s(2a+3b), then k must be even -/
theorem divisor_count_equality_implies_even (a b k : ℕ+) :
  k = s a ∧ k = s b ∧ k = s (2 * a + 3 * b) → Even k := by sorry

end NUMINAMATH_CALUDE_divisor_count_equality_implies_even_l75_7514


namespace NUMINAMATH_CALUDE_y_range_given_x_constraints_l75_7518

theorem y_range_given_x_constraints (x y : ℝ) 
  (h1 : 2 ≤ |x - 5| ∧ |x - 5| ≤ 9) 
  (h2 : y = 3 * x + 2) : 
  y ∈ Set.Icc (-10) 11 ∪ Set.Icc 23 44 := by
  sorry

end NUMINAMATH_CALUDE_y_range_given_x_constraints_l75_7518


namespace NUMINAMATH_CALUDE_solve_equation_l75_7566

theorem solve_equation : ∃ x : ℝ, 2.25 * x = 45 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l75_7566


namespace NUMINAMATH_CALUDE_x_equals_one_l75_7571

theorem x_equals_one (x y : ℕ+) 
  (h : ∀ n : ℕ+, (n * y)^2 + 1 ∣ x^(Nat.totient n) - 1) : 
  x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_l75_7571


namespace NUMINAMATH_CALUDE_roots_position_l75_7540

theorem roots_position (a b : ℝ) :
  ∃ (x₁ x₂ : ℝ), (x₁ - a) * (x₁ - a - b) = 1 ∧
                  (x₂ - a) * (x₂ - a - b) = 1 ∧
                  x₁ < a ∧ a < x₂ := by
  sorry

end NUMINAMATH_CALUDE_roots_position_l75_7540


namespace NUMINAMATH_CALUDE_true_discount_calculation_l75_7592

/-- Given the present worth and banker's gain, calculate the true discount -/
theorem true_discount_calculation (present_worth banker_gain : ℕ) 
  (h1 : present_worth = 576) 
  (h2 : banker_gain = 16) : 
  present_worth + banker_gain = 592 := by
  sorry

#check true_discount_calculation

end NUMINAMATH_CALUDE_true_discount_calculation_l75_7592


namespace NUMINAMATH_CALUDE_childrens_vehicle_wheels_l75_7511

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

end NUMINAMATH_CALUDE_childrens_vehicle_wheels_l75_7511


namespace NUMINAMATH_CALUDE_nabla_equation_solution_l75_7565

/-- The nabla operation defined for real numbers -/
def nabla (a b : ℝ) : ℝ := (a + 1) * (b - 2)

/-- Theorem: If 5 ∇ x = 30, then x = 7 -/
theorem nabla_equation_solution :
  ∀ x : ℝ, nabla 5 x = 30 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_nabla_equation_solution_l75_7565


namespace NUMINAMATH_CALUDE_octagon_circles_theorem_l75_7546

theorem octagon_circles_theorem (r : ℝ) (a b : ℤ) : 
  (∃ (s : ℝ), s = 2 ∧ s = r * Real.sqrt (2 - Real.sqrt 2)) →
  r^2 = a + b * Real.sqrt 2 →
  (a : ℝ) + b = 6 := by
sorry

end NUMINAMATH_CALUDE_octagon_circles_theorem_l75_7546


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l75_7563

theorem cost_price_of_ball (selling_price : ℕ) (num_balls : ℕ) (loss_balls : ℕ) :
  selling_price = 720 ∧ num_balls = 17 ∧ loss_balls = 5 →
  ∃ (cost_price : ℕ), cost_price * num_balls - cost_price * loss_balls = selling_price ∧ cost_price = 60 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l75_7563


namespace NUMINAMATH_CALUDE_jam_cost_is_348_l75_7585

/-- The cost of jam used for all sandwiches --/
def jam_cost (N B J : ℕ) : ℚ :=
  (N * J * 6 : ℕ) / 100

/-- The total cost of ingredients for all sandwiches --/
def total_cost (N B J : ℕ) : ℚ :=
  (N * (5 * B + 6 * J) : ℕ) / 100

theorem jam_cost_is_348 (N B J : ℕ) :
  N > 1 ∧ B > 0 ∧ J > 0 ∧ total_cost N B J = 348 / 100 → jam_cost N B J = 348 / 100 := by
  sorry

end NUMINAMATH_CALUDE_jam_cost_is_348_l75_7585


namespace NUMINAMATH_CALUDE_function_range_l75_7508

/-- Given a^2 - a < 2 and a is a positive integer, 
    the range of f(x) = x + 2a/x is (-∞, -2√2] ∪ [2√2, +∞) -/
theorem function_range (a : ℕ+) (h : a^2 - a < 2) :
  Set.range (fun x : ℝ => x + 2*a/x) = 
    Set.Iic (-2 * Real.sqrt 2) ∪ Set.Ici (2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_function_range_l75_7508


namespace NUMINAMATH_CALUDE_percentage_addition_l75_7506

theorem percentage_addition (x : ℝ) : x * 30 / 100 + 15 * 50 / 100 = 10.5 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_addition_l75_7506


namespace NUMINAMATH_CALUDE_bacon_suggestion_count_bacon_suggestion_proof_l75_7557

theorem bacon_suggestion_count : ℕ → ℕ → ℕ → Prop :=
  fun mashed_potatoes_count difference bacon_count =>
    (mashed_potatoes_count = 457) →
    (mashed_potatoes_count = bacon_count + difference) →
    (difference = 63) →
    (bacon_count = 394)

-- The proof is omitted
theorem bacon_suggestion_proof : bacon_suggestion_count 457 63 394 := by
  sorry

end NUMINAMATH_CALUDE_bacon_suggestion_count_bacon_suggestion_proof_l75_7557


namespace NUMINAMATH_CALUDE_twin_brothers_age_product_difference_l75_7503

theorem twin_brothers_age_product_difference :
  ∀ (current_age : ℕ),
  current_age = 4 →
  (current_age + 1) * (current_age + 1) - current_age * current_age = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_twin_brothers_age_product_difference_l75_7503


namespace NUMINAMATH_CALUDE_pairs_count_l75_7598

/-- S(n) denotes the sum of the digits of a natural number n -/
def S (n : ℕ) : ℕ := sorry

/-- The number of pairs <m, n> satisfying the given conditions -/
def count_pairs : ℕ := sorry

theorem pairs_count :
  count_pairs = 99 ∧
  ∀ m n : ℕ,
    m < 100 →
    n < 100 →
    m > n →
    m + S n = n + 2 * S m →
    (m, n) ∈ (Finset.filter (fun p : ℕ × ℕ => 
      p.1 < 100 ∧
      p.2 < 100 ∧
      p.1 > p.2 ∧
      p.1 + S p.2 = p.2 + 2 * S p.1)
    (Finset.product (Finset.range 100) (Finset.range 100))) :=
by sorry

end NUMINAMATH_CALUDE_pairs_count_l75_7598


namespace NUMINAMATH_CALUDE_meaningful_fraction_l75_7559

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (x + 3) / (x - 4)) ↔ x ≠ 4 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l75_7559


namespace NUMINAMATH_CALUDE_quadratic_passes_through_points_l75_7522

/-- A quadratic function passing through the points (2,0), (0,4), and (-2,0) -/
def quadratic_function (x : ℝ) : ℝ := -x^2 + 4

/-- Theorem stating that the quadratic function passes through the given points -/
theorem quadratic_passes_through_points :
  (quadratic_function 2 = 0) ∧
  (quadratic_function 0 = 4) ∧
  (quadratic_function (-2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_passes_through_points_l75_7522


namespace NUMINAMATH_CALUDE_statements_c_and_d_are_correct_l75_7597

theorem statements_c_and_d_are_correct :
  (∀ a b c : ℝ, c^2 > 0 → a*c^2 > b*c^2 → a > b) ∧
  (∀ a b m : ℝ, a > b → b > 0 → m > 0 → (b+m)/(a+m) > b/a) :=
by sorry

end NUMINAMATH_CALUDE_statements_c_and_d_are_correct_l75_7597


namespace NUMINAMATH_CALUDE_rotation_and_inclination_l75_7531

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

end NUMINAMATH_CALUDE_rotation_and_inclination_l75_7531


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l75_7599

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) :
  a = 6 →
  c = 4 →
  Real.sin (B / 2) = Real.sqrt 3 / 3 →
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l75_7599


namespace NUMINAMATH_CALUDE_simplify_expression_l75_7525

theorem simplify_expression (a : ℝ) : 
  (1/2) * (8 * a^2 + 4 * a) - 3 * (a - (1/3) * a^2) = 5 * a^2 - a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l75_7525


namespace NUMINAMATH_CALUDE_notebook_cost_l75_7567

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

end NUMINAMATH_CALUDE_notebook_cost_l75_7567


namespace NUMINAMATH_CALUDE_lilith_cap_collection_l75_7516

/-- Calculates the number of caps Lilith has collected after a given number of years -/
def caps_collected (years : ℕ) : ℕ :=
  let first_year_caps := 3 * 12
  let subsequent_years_caps := 5 * 12 * (years - 1)
  let christmas_caps := 40 * years
  let total_caps := first_year_caps + subsequent_years_caps + christmas_caps
  let lost_caps := 15 * years
  total_caps - lost_caps

/-- Theorem stating that Lilith has collected 401 caps after 5 years -/
theorem lilith_cap_collection : caps_collected 5 = 401 := by
  sorry

end NUMINAMATH_CALUDE_lilith_cap_collection_l75_7516


namespace NUMINAMATH_CALUDE_cubic_root_function_l75_7517

/-- Given a function y = kx^(1/3) where y = 5√2 when x = 64, 
    prove that y = 2.5√2 when x = 8 -/
theorem cubic_root_function (k : ℝ) : 
  (∃ y : ℝ, y = k * 64^(1/3) ∧ y = 5 * Real.sqrt 2) →
  k * 8^(1/3) = 2.5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_cubic_root_function_l75_7517


namespace NUMINAMATH_CALUDE_alicia_book_cost_l75_7521

/-- The total cost of books given the number of each type and their individual costs -/
def total_cost (math_books art_books science_books : ℕ) (math_cost art_cost science_cost : ℕ) : ℕ :=
  math_books * math_cost + art_books * art_cost + science_books * science_cost

/-- Theorem stating that the total cost of Alicia's books is $30 -/
theorem alicia_book_cost : total_cost 2 3 6 3 2 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_alicia_book_cost_l75_7521


namespace NUMINAMATH_CALUDE_f_g_one_eq_one_solution_set_eq_two_l75_7555

-- Define the domain of x
inductive X : Type
| one : X
| two : X
| three : X

-- Define functions f and g
def f : X → ℕ
| X.one => 1
| X.two => 3
| X.three => 1

def g : X → ℕ
| X.one => 3
| X.two => 2
| X.three => 1

-- Define composition of f and g
def f_comp_g (x : X) : ℕ := f (match g x with
  | 1 => X.one
  | 2 => X.two
  | 3 => X.three
  | _ => X.one)

def g_comp_f (x : X) : ℕ := g (match f x with
  | 1 => X.one
  | 2 => X.two
  | 3 => X.three
  | _ => X.one)

theorem f_g_one_eq_one : f_comp_g X.one = 1 := by sorry

theorem solution_set_eq_two :
  (∀ x : X, f_comp_g x > g_comp_f x ↔ x = X.two) := by sorry

end NUMINAMATH_CALUDE_f_g_one_eq_one_solution_set_eq_two_l75_7555


namespace NUMINAMATH_CALUDE_square_neq_four_implies_neq_two_l75_7586

theorem square_neq_four_implies_neq_two (a : ℝ) :
  (a^2 ≠ 4 → a ≠ 2) ∧ ¬(∀ a : ℝ, a ≠ 2 → a^2 ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_square_neq_four_implies_neq_two_l75_7586


namespace NUMINAMATH_CALUDE_inequality_proof_l75_7587

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l75_7587


namespace NUMINAMATH_CALUDE_prob_king_or_queen_is_two_thirteenths_l75_7505

-- Define the properties of a standard deck
structure StandardDeck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)
  (h_total : total_cards = 52)
  (h_ranks : num_ranks = 13)
  (h_suits : num_suits = 4)
  (h_kings : num_kings = 4)
  (h_queens : num_queens = 4)
  (h_cards_per_rank : total_cards = num_ranks * num_suits)

-- Define the probability function
def probability_king_or_queen (deck : StandardDeck) : ℚ :=
  (deck.num_kings + deck.num_queens : ℚ) / deck.total_cards

-- State the theorem
theorem prob_king_or_queen_is_two_thirteenths (deck : StandardDeck) :
  probability_king_or_queen deck = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_or_queen_is_two_thirteenths_l75_7505


namespace NUMINAMATH_CALUDE_sand_tank_mass_l75_7528

/-- Given a tank filled with sand, prove that the total mass when completely filled
    is (8p - 3q) / 5, where p is the mass when 3/4 filled and q is the mass when 1/3 filled. -/
theorem sand_tank_mass (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p > q) :
  ∃ (z w : ℝ), z > 0 ∧ w > 0 ∧
    z + 3/4 * w = p ∧
    z + 1/3 * w = q ∧
    z + w = (8*p - 3*q) / 5 :=
by sorry

end NUMINAMATH_CALUDE_sand_tank_mass_l75_7528


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l75_7590

def initial_money : ℕ := 56
def notebooks_bought : ℕ := 7
def books_bought : ℕ := 2
def notebook_cost : ℕ := 4
def book_cost : ℕ := 7

theorem money_left_after_purchase : 
  initial_money - (notebooks_bought * notebook_cost + books_bought * book_cost) = 14 :=
by sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l75_7590


namespace NUMINAMATH_CALUDE_allison_wins_prob_l75_7530

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

end NUMINAMATH_CALUDE_allison_wins_prob_l75_7530


namespace NUMINAMATH_CALUDE_anthony_transaction_percentage_l75_7554

theorem anthony_transaction_percentage (mabel_transactions cal_transactions jade_transactions : ℕ)
  (anthony_transactions : ℕ) :
  mabel_transactions = 90 →
  cal_transactions = (2 : ℚ) / 3 * anthony_transactions →
  jade_transactions = cal_transactions + 14 →
  jade_transactions = 80 →
  (anthony_transactions - mabel_transactions : ℚ) / mabel_transactions * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_anthony_transaction_percentage_l75_7554


namespace NUMINAMATH_CALUDE_min_marked_elements_eq_666_l75_7558

/-- The minimum number of marked elements in {1, ..., 2000} such that
    for every pair (k, 2k) where 1 ≤ k ≤ 1000, at least one of k or 2k is marked. -/
def min_marked_elements : ℕ :=
  let S := Finset.range 2000
  Finset.filter (fun n => ∃ k ∈ Finset.range 1000, n = k ∨ n = 2 * k) S |>.card

/-- The theorem stating that the minimum number of marked elements is 666. -/
theorem min_marked_elements_eq_666 : min_marked_elements = 666 := by
  sorry

end NUMINAMATH_CALUDE_min_marked_elements_eq_666_l75_7558


namespace NUMINAMATH_CALUDE_probability_three_odd_dice_l75_7534

theorem probability_three_odd_dice (n : ℕ) (p : ℝ) : 
  n = 5 →                          -- number of dice
  p = 1 / 2 →                      -- probability of rolling an odd number on a single die
  (Nat.choose n 3 : ℝ) * p^3 * (1 - p)^(n - 3) = 5 / 16 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_odd_dice_l75_7534


namespace NUMINAMATH_CALUDE_line_equation_slope_intercept_l75_7519

/-- The equation of a line with slope -1 and y-intercept -1 is x + y + 1 = 0 -/
theorem line_equation_slope_intercept (x y : ℝ) : 
  (∀ x y, y = -x - 1) ↔ (∀ x y, x + y + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_equation_slope_intercept_l75_7519


namespace NUMINAMATH_CALUDE_probability_of_valid_pair_l75_7593

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

end NUMINAMATH_CALUDE_probability_of_valid_pair_l75_7593


namespace NUMINAMATH_CALUDE_triangle_side_length_l75_7580

theorem triangle_side_length (a b c : ℝ) : 
  a = 1 → b = 3 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  c ∈ ({3, 4, 5, 6} : Set ℝ) →
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l75_7580


namespace NUMINAMATH_CALUDE_isosceles_triangle_figure_triangle_count_l75_7581

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

end NUMINAMATH_CALUDE_isosceles_triangle_figure_triangle_count_l75_7581


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_fractional_part_smallest_cube_root_exists_smallest_cube_root_is_68922_l75_7515

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

end NUMINAMATH_CALUDE_smallest_cube_root_with_fractional_part_smallest_cube_root_exists_smallest_cube_root_is_68922_l75_7515


namespace NUMINAMATH_CALUDE_unique_c_for_unique_quadratic_solution_l75_7532

theorem unique_c_for_unique_quadratic_solution :
  ∃! (c : ℝ), c ≠ 0 ∧
  (∃! (b : ℝ), b > 0 ∧
    (∃! (x : ℝ), x^2 + (b^2 + 1/b^2) * x + c = 0)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_c_for_unique_quadratic_solution_l75_7532


namespace NUMINAMATH_CALUDE_total_money_after_redistribution_l75_7573

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

end NUMINAMATH_CALUDE_total_money_after_redistribution_l75_7573


namespace NUMINAMATH_CALUDE_milk_fraction_after_transfers_l75_7584

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

end NUMINAMATH_CALUDE_milk_fraction_after_transfers_l75_7584


namespace NUMINAMATH_CALUDE_defeated_candidate_percentage_approx_l75_7510

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

end NUMINAMATH_CALUDE_defeated_candidate_percentage_approx_l75_7510


namespace NUMINAMATH_CALUDE_sin_315_degrees_l75_7529

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l75_7529


namespace NUMINAMATH_CALUDE_grapefruit_orchards_count_l75_7542

/-- Calculates the number of grapefruit orchards in a citrus grove. -/
def grapefruit_orchards (total : ℕ) (lemon : ℕ) : ℕ :=
  let orange := lemon / 2
  let remaining := total - (lemon + orange)
  remaining / 2

/-- Proves that the number of grapefruit orchards is 2 given the specified conditions. -/
theorem grapefruit_orchards_count :
  grapefruit_orchards 16 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_grapefruit_orchards_count_l75_7542


namespace NUMINAMATH_CALUDE_attendance_difference_l75_7556

def football_game_attendance (saturday_attendance : ℕ) : Prop :=
  let monday_attendance : ℕ := saturday_attendance - saturday_attendance / 4
  let wednesday_attendance : ℕ := monday_attendance + monday_attendance / 2
  let friday_attendance : ℕ := saturday_attendance + monday_attendance
  let thursday_attendance : ℕ := 45
  let sunday_attendance : ℕ := saturday_attendance - saturday_attendance * 15 / 100
  let total_attendance : ℕ := saturday_attendance + monday_attendance + wednesday_attendance + 
                               thursday_attendance + friday_attendance + sunday_attendance
  let expected_attendance : ℕ := 350
  total_attendance - expected_attendance = 133

theorem attendance_difference : 
  football_game_attendance 80 :=
sorry

end NUMINAMATH_CALUDE_attendance_difference_l75_7556


namespace NUMINAMATH_CALUDE_dave_ice_cubes_l75_7537

/-- Given that Dave started with 2 ice cubes and ended with 9 ice cubes in total,
    prove that he made 7 additional ice cubes. -/
theorem dave_ice_cubes (initial : Nat) (final : Nat) (h1 : initial = 2) (h2 : final = 9) :
  final - initial = 7 := by
  sorry

end NUMINAMATH_CALUDE_dave_ice_cubes_l75_7537


namespace NUMINAMATH_CALUDE_choose_two_correct_l75_7548

/-- The number of ways to choose 2 different items from n distinct items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that choose_two gives the correct number of ways to choose 2 from n -/
theorem choose_two_correct (n : ℕ) : choose_two n = Nat.choose n 2 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_correct_l75_7548


namespace NUMINAMATH_CALUDE_probability_yellow_ball_l75_7560

/-- The probability of drawing a yellow ball from a bag containing white and yellow balls -/
theorem probability_yellow_ball (total_balls : ℕ) (white_balls yellow_balls : ℕ) 
  (h1 : total_balls = white_balls + yellow_balls)
  (h2 : total_balls > 0)
  (h3 : white_balls = 2)
  (h4 : yellow_balls = 3) :
  (yellow_balls : ℚ) / total_balls = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_yellow_ball_l75_7560


namespace NUMINAMATH_CALUDE_cloth_profit_theorem_l75_7502

/-- Calculates the profit per meter of cloth given the total meters sold, 
    total selling price, and cost price per meter. -/
def profit_per_meter (meters_sold : ℕ) (total_selling_price : ℚ) (cost_price_per_meter : ℚ) : ℚ :=
  (total_selling_price - (meters_sold : ℚ) * cost_price_per_meter) / (meters_sold : ℚ)

/-- Theorem stating that given 85 meters of cloth sold for $8925 
    with a cost price of $90 per meter, the profit per meter is $15. -/
theorem cloth_profit_theorem :
  profit_per_meter 85 8925 90 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cloth_profit_theorem_l75_7502


namespace NUMINAMATH_CALUDE_sum_of_odd_powers_divisible_by_61_l75_7500

theorem sum_of_odd_powers_divisible_by_61 
  (a₁ a₂ a₃ a₄ : ℤ) 
  (h : a₁^3 + a₂^3 + a₃^3 + a₄^3 = 0) :
  ∀ k : ℕ, k % 2 = 1 → k > 0 → 
  (61 : ℤ) ∣ (a₁^k + a₂^k + a₃^k + a₄^k) := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_powers_divisible_by_61_l75_7500


namespace NUMINAMATH_CALUDE_guitar_payment_plan_l75_7536

theorem guitar_payment_plan (total_with_interest : ℝ) (num_months : ℕ) (interest_rate : ℝ) :
  total_with_interest = 1320 →
  num_months = 12 →
  interest_rate = 0.1 →
  ∃ (monthly_payment : ℝ),
    monthly_payment * num_months * (1 + interest_rate) = total_with_interest ∧
    monthly_payment = 100 := by
  sorry

end NUMINAMATH_CALUDE_guitar_payment_plan_l75_7536


namespace NUMINAMATH_CALUDE_absolute_value_32_l75_7507

theorem absolute_value_32 (x : ℝ) : |x| = 32 → x = 32 ∨ x = -32 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_32_l75_7507


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l75_7576

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ,
    x ≤ y →
    x^2 + y^2 = 3 * 2016^z + 77 →
    ((x = 4 ∧ y = 8 ∧ z = 0) ∨
     (x = 14 ∧ y = 77 ∧ z = 1) ∨
     (x = 35 ∧ y = 70 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l75_7576


namespace NUMINAMATH_CALUDE_joyce_initial_apples_l75_7535

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

end NUMINAMATH_CALUDE_joyce_initial_apples_l75_7535


namespace NUMINAMATH_CALUDE_roots_product_theorem_l75_7574

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

end NUMINAMATH_CALUDE_roots_product_theorem_l75_7574


namespace NUMINAMATH_CALUDE_largest_interesting_is_correct_l75_7545

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

end NUMINAMATH_CALUDE_largest_interesting_is_correct_l75_7545


namespace NUMINAMATH_CALUDE_worker_arrival_time_l75_7572

/-- Proves that a worker walking at 4/5 of her normal speed arrives 10 minutes later -/
theorem worker_arrival_time (normal_speed : ℝ) (normal_time : ℝ) 
  (h1 : normal_time = 40)
  (h2 : normal_speed > 0) :
  let reduced_speed := (4/5 : ℝ) * normal_speed
  let new_time := normal_time * (normal_speed / reduced_speed)
  new_time - normal_time = 10 := by
sorry


end NUMINAMATH_CALUDE_worker_arrival_time_l75_7572


namespace NUMINAMATH_CALUDE_apple_difference_l75_7596

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

end NUMINAMATH_CALUDE_apple_difference_l75_7596


namespace NUMINAMATH_CALUDE_function_and_triangle_properties_l75_7550

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

end NUMINAMATH_CALUDE_function_and_triangle_properties_l75_7550


namespace NUMINAMATH_CALUDE_blocks_left_l75_7583

theorem blocks_left (initial_blocks : ℕ) (used_blocks : ℕ) : 
  initial_blocks = 78 → used_blocks = 19 → initial_blocks - used_blocks = 59 := by
sorry

end NUMINAMATH_CALUDE_blocks_left_l75_7583


namespace NUMINAMATH_CALUDE_distance_to_midpoint_l75_7512

/-- The distance from Shinyoung's house to the midpoint of the path to school -/
theorem distance_to_midpoint (house_to_office village_to_school : ℕ) : 
  house_to_office = 1700 →
  village_to_school = 900 →
  (house_to_office + village_to_school) / 2 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_midpoint_l75_7512


namespace NUMINAMATH_CALUDE_sum_of_integers_l75_7552

theorem sum_of_integers (m n : ℕ+) 
  (h1 : m^2 + n^2 = 3789)
  (h2 : Nat.gcd m.val n.val + Nat.lcm m.val n.val = 633) : 
  m + n = 87 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l75_7552


namespace NUMINAMATH_CALUDE_red_balls_count_l75_7523

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

end NUMINAMATH_CALUDE_red_balls_count_l75_7523


namespace NUMINAMATH_CALUDE_museum_visit_l75_7541

theorem museum_visit (num_students : ℕ) (ticket_price : ℕ) :
  (∃ k : ℕ, num_students = 5 * k) →
  (num_students + 1) * (ticket_price / 2) = 1599 →
  ticket_price % 2 = 0 →
  num_students = 40 ∧ ticket_price = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_museum_visit_l75_7541


namespace NUMINAMATH_CALUDE_randys_trip_length_l75_7594

theorem randys_trip_length :
  ∀ (x : ℚ),
  (x / 4 : ℚ) + 40 + 10 + (x / 6 : ℚ) = x →
  x = 600 / 7 := by
sorry

end NUMINAMATH_CALUDE_randys_trip_length_l75_7594


namespace NUMINAMATH_CALUDE_product_inequality_l75_7504

theorem product_inequality (a b c d : ℝ) : a > b ∧ b > 0 ∧ c > d ∧ d > 0 → a * c > b * d := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l75_7504


namespace NUMINAMATH_CALUDE_cone_volume_maximization_l75_7547

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

end NUMINAMATH_CALUDE_cone_volume_maximization_l75_7547


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l75_7524

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

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l75_7524


namespace NUMINAMATH_CALUDE_race_participants_l75_7564

theorem race_participants (first_year : ℕ) (second_year : ℕ) : 
  first_year = 8 →
  second_year = 5 * first_year →
  first_year + second_year = 48 := by
  sorry

end NUMINAMATH_CALUDE_race_participants_l75_7564


namespace NUMINAMATH_CALUDE_maximize_product_l75_7544

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 28) :
  x^5 * y^3 ≤ 17.5^5 * 10.5^3 ∧
  (x^5 * y^3 = 17.5^5 * 10.5^3 ↔ x = 17.5 ∧ y = 10.5) :=
by sorry

end NUMINAMATH_CALUDE_maximize_product_l75_7544


namespace NUMINAMATH_CALUDE_absolute_difference_equation_l75_7553

theorem absolute_difference_equation : 
  ∃! x : ℝ, |16 - x| - |x - 12| = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_equation_l75_7553


namespace NUMINAMATH_CALUDE_fraction_multiplication_l75_7543

theorem fraction_multiplication (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (6 * x * y) / (5 * z^2) * (10 * z^3) / (9 * x * y) = (4 * z) / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l75_7543
