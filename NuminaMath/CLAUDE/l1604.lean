import Mathlib

namespace complex_division_simplification_l1604_160427

theorem complex_division_simplification :
  (2 - I) / (1 - I) = 3/2 + I/2 := by
  sorry

end complex_division_simplification_l1604_160427


namespace product_of_cosines_l1604_160449

theorem product_of_cosines : 
  (1 + Real.cos (π / 12)) * (1 + Real.cos (5 * π / 12)) * 
  (1 + Real.cos (7 * π / 12)) * (1 + Real.cos (11 * π / 12)) = 1 / 16 := by
  sorry

end product_of_cosines_l1604_160449


namespace min_sum_with_constraint_min_sum_achieved_l1604_160443

/-- Given two natural numbers a and b satisfying 1a + 4b = 30,
    their sum is minimized when a = b = 6 -/
theorem min_sum_with_constraint (a b : ℕ) (h : a + 4 * b = 30) :
  a + b ≥ 12 := by
sorry

/-- The minimum sum of 12 is achieved when a = b = 6 -/
theorem min_sum_achieved : ∃ (a b : ℕ), a + 4 * b = 30 ∧ a + b = 12 := by
sorry

end min_sum_with_constraint_min_sum_achieved_l1604_160443


namespace quadratic_roots_property_l1604_160428

theorem quadratic_roots_property (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + a*x₁ + 8 = 0 ∧ 
    x₂^2 + a*x₂ + 8 = 0 ∧ 
    x₁ - 64/(17*x₂^3) = x₂ - 64/(17*x₁^3)) 
  → a = 12 ∨ a = -12 :=
sorry

end quadratic_roots_property_l1604_160428


namespace income_comparison_l1604_160453

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.7)
  (h2 : mary = juan * 1.12) :
  (mary - tim) / tim = 0.6 :=
sorry

end income_comparison_l1604_160453


namespace f_negative_one_value_l1604_160477

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_negative_one_value : 
  (∀ x, f (x / (1 + x)) = x) → f (-1) = -1/2 := by
  sorry

end f_negative_one_value_l1604_160477


namespace sports_stars_arrangement_l1604_160491

/-- The number of ways to arrange players from multiple teams in a row, where teammates must sit together -/
def arrangement_count (team_sizes : List Nat) : Nat :=
  (Nat.factorial team_sizes.length) * (team_sizes.map Nat.factorial).prod

/-- Theorem: The number of ways to arrange 10 players from 4 teams (with 3, 3, 2, and 2 players respectively) in a row, where teammates must sit together, is 3456 -/
theorem sports_stars_arrangement :
  arrangement_count [3, 3, 2, 2] = 3456 := by
  sorry

#eval arrangement_count [3, 3, 2, 2]

end sports_stars_arrangement_l1604_160491


namespace linear_function_not_in_third_quadrant_l1604_160403

-- Define the linear function
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Define the condition for a point to be in the third quadrant
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem statement
theorem linear_function_not_in_third_quadrant 
  (k b : ℝ) 
  (h : ∀ x y : ℝ, y = linear_function k b x → ¬in_third_quadrant x y) : 
  k < 0 ∧ b ≥ 0 := by
  sorry

end linear_function_not_in_third_quadrant_l1604_160403


namespace open_box_volume_l1604_160485

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_length : ℝ) 
  (h1 : sheet_length = 48) 
  (h2 : sheet_width = 36) 
  (h3 : cut_length = 8) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 5120 := by
  sorry

#check open_box_volume

end open_box_volume_l1604_160485


namespace sufficient_not_necessary_parallel_l1604_160416

variable {V : Type*} [NormedAddCommGroup V] [Module ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem sufficient_not_necessary_parallel (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b : V, a + b = 0 → parallel a b) ∧
  (∃ a b : V, parallel a b ∧ a + b ≠ 0) :=
sorry

end sufficient_not_necessary_parallel_l1604_160416


namespace friendly_number_F_formula_max_friendly_N_l1604_160404

def is_friendly_number (M : ℕ) : Prop :=
  ∃ (a b c d : ℕ), M = 1000 * a + 100 * b + 10 * c + d ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ a - b = c - d

def F (M : ℕ) : ℤ :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  100 * a - 100 * b - 10 * b + c - d

theorem friendly_number_F_formula (M : ℕ) (h : is_friendly_number M) :
  F M = 100 * (M / 1000) - 110 * ((M / 100) % 10) + (M / 10) % 10 - M % 10 :=
sorry

def N (x y m n : ℕ) : ℕ := 1000 * x + 100 * y + 30 * m + n + 1001

theorem max_friendly_N (x y m n : ℕ) 
  (h1 : 0 ≤ y ∧ y < x ∧ x ≤ 8) 
  (h2 : 0 ≤ m ∧ m ≤ 3) 
  (h3 : 0 ≤ n ∧ n ≤ 8) 
  (h4 : is_friendly_number (N x y m n)) 
  (h5 : F (N x y m n) % 5 = 1) :
  N x y m n ≤ 9696 :=
sorry

end friendly_number_F_formula_max_friendly_N_l1604_160404


namespace jayda_spending_l1604_160487

theorem jayda_spending (aitana_spending jayda_spending : ℚ) : 
  aitana_spending = jayda_spending + (2/5 : ℚ) * jayda_spending →
  aitana_spending + jayda_spending = 960 →
  jayda_spending = 400 :=
by
  sorry

end jayda_spending_l1604_160487


namespace olivia_spent_15_dollars_l1604_160447

/-- The amount spent at a supermarket, given the initial amount and the amount left after spending. -/
def amount_spent (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

/-- Proves that Olivia spent 15 dollars at the supermarket. -/
theorem olivia_spent_15_dollars : amount_spent 78 63 = 15 := by
  sorry

end olivia_spent_15_dollars_l1604_160447


namespace prob_two_black_one_red_standard_deck_l1604_160401

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    red_cards := 26,
    black_cards := 26 }

/-- The probability of drawing two black cards followed by a red card -/
def prob_two_black_one_red (d : Deck) : ℚ :=
  (d.black_cards * (d.black_cards - 1) * d.red_cards) / 
  (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2))

/-- Theorem stating the probability for a standard deck -/
theorem prob_two_black_one_red_standard_deck :
  prob_two_black_one_red standard_deck = 13 / 102 := by
  sorry

end prob_two_black_one_red_standard_deck_l1604_160401


namespace rectangle_cutout_equals_square_area_l1604_160499

theorem rectangle_cutout_equals_square_area : 
  (10 * 7 - 1 * 6 : ℕ) = 8 * 8 := by sorry

end rectangle_cutout_equals_square_area_l1604_160499


namespace flower_count_proof_l1604_160410

theorem flower_count_proof (total : ℕ) (red green blue yellow purple orange : ℕ) : 
  total = 180 →
  red = (30 * total) / 100 →
  green = (10 * total) / 100 →
  blue = green / 2 →
  yellow = red + 5 →
  3 * purple = 7 * orange →
  red + green + blue + yellow + purple + orange = total →
  red = 54 ∧ green = 18 ∧ blue = 9 ∧ yellow = 59 ∧ purple = 12 ∧ orange = 28 :=
by sorry

end flower_count_proof_l1604_160410


namespace heejin_is_oldest_l1604_160437

-- Define the ages of the three friends
def yoona_age : ℕ := 23
def miyoung_age : ℕ := 22
def heejin_age : ℕ := 24

-- Theorem stating that Heejin is the oldest
theorem heejin_is_oldest : 
  heejin_age ≥ yoona_age ∧ heejin_age ≥ miyoung_age := by
  sorry

end heejin_is_oldest_l1604_160437


namespace odd_function_condition_l1604_160463

def f (x a b : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x, f (-x) a b = -f x a b) ↔ a^2 + b^2 = 0 := by sorry

end odd_function_condition_l1604_160463


namespace least_positive_integer_with_remainders_l1604_160482

theorem least_positive_integer_with_remainders : ∃ (M : ℕ), 
  (M > 0) ∧
  (M % 6 = 5) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (M % 11 = 10) ∧
  (M % 12 = 11) ∧
  (M % 13 = 12) ∧
  (∀ (N : ℕ), N > 0 ∧ 
    N % 6 = 5 ∧
    N % 8 = 7 ∧
    N % 9 = 8 ∧
    N % 11 = 10 ∧
    N % 12 = 11 ∧
    N % 13 = 12 → M ≤ N) ∧
  M = 10163 :=
by sorry

end least_positive_integer_with_remainders_l1604_160482


namespace cube_sum_equals_linear_sum_l1604_160479

theorem cube_sum_equals_linear_sum (a b : ℝ) 
  (h : a / (1 + b) + b / (1 + a) = 1) : 
  a^3 + b^3 = a + b := by
  sorry

end cube_sum_equals_linear_sum_l1604_160479


namespace quadratic_real_roots_range_l1604_160474

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0) → m ≤ 1 := by
  sorry

end quadratic_real_roots_range_l1604_160474


namespace shoe_selection_probability_l1604_160460

def total_pairs : ℕ := 16
def black_pairs : ℕ := 8
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 2
def red_pairs : ℕ := 2

theorem shoe_selection_probability :
  let total_shoes := total_pairs * 2
  let prob_same_color_diff_foot : ℚ :=
    (black_pairs * black_pairs + brown_pairs * brown_pairs + 
     gray_pairs * gray_pairs + red_pairs * red_pairs) / 
    (total_shoes * (total_shoes - 1))
  prob_same_color_diff_foot = 11 / 62 := by sorry

end shoe_selection_probability_l1604_160460


namespace three_roles_four_people_l1604_160484

def number_of_assignments (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / Nat.factorial (n - k)

theorem three_roles_four_people :
  number_of_assignments 4 3 = 24 :=
by
  sorry

end three_roles_four_people_l1604_160484


namespace length_PS_l1604_160467

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  ∃ (x y z : ℝ), P = (0, 0) ∧ Q = (x, y) ∧ R = (z, 0)

-- Define a right angle at P
def RightAngleAtP (P Q R : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0

-- Define the lengths of PR and PQ
def LengthPR (P R : ℝ × ℝ) : ℝ := 3
def LengthPQ (P Q : ℝ × ℝ) : ℝ := 4

-- Define S as the point where the angle bisector of ∠QPR meets QR
def AngleBisectorS (P Q R S : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ S = (t * Q.1 + (1 - t) * R.1, t * Q.2 + (1 - t) * R.2) ∧
  (S.1 - P.1) * (Q.2 - P.2) = (S.2 - P.2) * (Q.1 - P.1) ∧
  (S.1 - P.1) * (R.2 - P.2) = (S.2 - P.2) * (R.1 - P.1)

-- Main theorem
theorem length_PS (P Q R S : ℝ × ℝ) :
  Triangle P Q R →
  RightAngleAtP P Q R →
  LengthPR P R = 3 →
  LengthPQ P Q = 4 →
  AngleBisectorS P Q R S →
  Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) = 20/7 := by
  sorry

end length_PS_l1604_160467


namespace initial_hats_count_l1604_160494

/-- Represents the state of hat distribution among gentlemen -/
structure HatDistribution where
  total : Nat
  withHat : Nat
  withoutHat : Nat
  givenMoreThanReceived : Nat

/-- Theorem stating that if 10 out of 20 gentlemen gave away more hats than they received,
    then the initial number of gentlemen with hats must be 10 -/
theorem initial_hats_count (dist : HatDistribution) :
  dist.total = 20 ∧
  dist.givenMoreThanReceived = 10 ∧
  dist.withHat + dist.withoutHat = dist.total →
  dist.withHat = 10 := by
  sorry

end initial_hats_count_l1604_160494


namespace product_of_three_numbers_l1604_160431

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 22)
  (sum_squares_eq : a^2 + b^2 + c^2 = 404)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 9346) :
  a * b * c = 446 := by
sorry

end product_of_three_numbers_l1604_160431


namespace farah_order_match_sticks_l1604_160445

/-- The number of boxes Farah ordered -/
def num_boxes : ℕ := 4

/-- The number of matchboxes in each box -/
def matchboxes_per_box : ℕ := 20

/-- The number of sticks in each matchbox -/
def sticks_per_matchbox : ℕ := 300

/-- The total number of match sticks Farah ordered -/
def total_match_sticks : ℕ := num_boxes * matchboxes_per_box * sticks_per_matchbox

theorem farah_order_match_sticks :
  total_match_sticks = 24000 := by
  sorry

end farah_order_match_sticks_l1604_160445


namespace largest_domain_of_g_l1604_160405

def is_valid_domain (S : Set ℝ) (g : ℝ → ℝ) : Prop :=
  ∀ x ∈ S, x^2 ∈ S ∧ 1/x^2 ∈ S ∧ g x + g (1/x^2) = x^2

theorem largest_domain_of_g :
  ∃! S : Set ℝ, is_valid_domain S g ∧
    ∀ T : Set ℝ, is_valid_domain T g → T ⊆ S :=
by
  sorry

#check largest_domain_of_g

end largest_domain_of_g_l1604_160405


namespace wall_width_proof_l1604_160492

def wall_height : ℝ := 4
def wall_area : ℝ := 16

theorem wall_width_proof :
  ∃ (width : ℝ), width * wall_height = wall_area ∧ width = 4 := by
sorry

end wall_width_proof_l1604_160492


namespace A_intersect_B_empty_l1604_160488

def A : Set ℤ := {x | ∃ n : ℕ+, x = 2*n - 1}

def B : Set ℤ := {y | ∃ x ∈ A, y = 3*x - 1}

theorem A_intersect_B_empty : A ∩ B = ∅ := by
  sorry

end A_intersect_B_empty_l1604_160488


namespace range_of_m_l1604_160495

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y - x / Real.exp 1) * (Real.log x - Real.log y) - y / m ≤ 0) → 
  0 < m ∧ m ≤ 1 := by
sorry

end range_of_m_l1604_160495


namespace fraction_inequality_l1604_160459

theorem fraction_inequality (x : ℝ) : x / (x + 1) < 0 ↔ -1 < x ∧ x < 0 := by sorry

end fraction_inequality_l1604_160459


namespace maximal_regions_quadrilaterals_l1604_160464

/-- The maximal number of regions created by n convex quadrilaterals in a plane -/
def maxRegions (n : ℕ) : ℕ := 4*n^2 - 4*n + 2

/-- Theorem stating that maxRegions gives the maximal number of regions -/
theorem maximal_regions_quadrilaterals (n : ℕ) :
  ∀ (regions : ℕ), regions ≤ maxRegions n :=
by sorry

end maximal_regions_quadrilaterals_l1604_160464


namespace dinner_cost_theorem_l1604_160446

/-- Calculate the total amount Bret spends on dinner -/
def dinner_cost : ℝ :=
  let team_a_size : ℕ := 4
  let team_b_size : ℕ := 4
  let main_meal_cost : ℝ := 12.00
  let team_a_appetizers : ℕ := 2
  let team_a_appetizer_cost : ℝ := 6.00
  let team_b_appetizers : ℕ := 3
  let team_b_appetizer_cost : ℝ := 8.00
  let sharing_plates : ℕ := 4
  let sharing_plate_cost : ℝ := 10.00
  let tip_percentage : ℝ := 0.20
  let rush_order_fee : ℝ := 5.00
  let sales_tax_rate : ℝ := 0.07

  let main_meals_cost := (team_a_size + team_b_size) * main_meal_cost
  let appetizers_cost := team_a_appetizers * team_a_appetizer_cost + team_b_appetizers * team_b_appetizer_cost
  let sharing_plates_cost := sharing_plates * sharing_plate_cost
  let food_cost := main_meals_cost + appetizers_cost + sharing_plates_cost
  let tip := food_cost * tip_percentage
  let subtotal := food_cost + tip + rush_order_fee
  let sales_tax := (food_cost + tip) * sales_tax_rate
  food_cost + tip + rush_order_fee + sales_tax

theorem dinner_cost_theorem : dinner_cost = 225.85 := by
  sorry

end dinner_cost_theorem_l1604_160446


namespace algebraic_expression_equality_l1604_160473

theorem algebraic_expression_equality (a b : ℝ) (h : a - 3 * b = -3) : 
  5 - a + 3 * b = 8 := by
  sorry

end algebraic_expression_equality_l1604_160473


namespace business_trip_distance_l1604_160421

/-- Calculates the total distance traveled during a business trip -/
theorem business_trip_distance (total_duration : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_duration = 8 →
  speed1 = 70 →
  speed2 = 85 →
  (total_duration / 2 * speed1) + (total_duration / 2 * speed2) = 620 := by
  sorry

#check business_trip_distance

end business_trip_distance_l1604_160421


namespace line_parameterization_l1604_160478

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 5 * x - 7

/-- The parametric form of the line -/
def parametric_form (s m t x y : ℝ) : Prop :=
  x = s + 2 * t ∧ y = 3 + m * t

/-- The theorem stating that s = 2 and m = 10 for the given line and parametric form -/
theorem line_parameterization :
  ∃ (s m : ℝ), (∀ (x y t : ℝ), line_equation x y ↔ parametric_form s m t x y) ∧ s = 2 ∧ m = 10 := by
  sorry


end line_parameterization_l1604_160478


namespace min_value_of_3a_plus_2_l1604_160433

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 4 * a^2 + 7 * a + 3 = 2) :
  ∃ (m : ℝ), m = 3 * a + 2 ∧ ∀ (x : ℝ), (4 * x^2 + 7 * x + 3 = 2) → m ≤ 3 * x + 2 ∧ m = -1 := by
  sorry

end min_value_of_3a_plus_2_l1604_160433


namespace sum_of_qp_values_l1604_160461

def p (x : ℝ) : ℝ := |x| + 1

def q (x : ℝ) : ℝ := -|x - 1|

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3]

theorem sum_of_qp_values :
  (x_values.map (λ x => q (p x))).sum = -21 := by sorry

end sum_of_qp_values_l1604_160461


namespace range_of_a_l1604_160458

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - x + 2 * a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, (∃ x : ℝ, f a x = y) ↔ (∃ x : ℝ, f a (f a x) = y)) →
  a ∈ Set.Ioo (1/2) 1 :=
sorry

end range_of_a_l1604_160458


namespace locus_of_G_is_parabola_l1604_160402

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a right triangle -/
structure RightTriangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Finds the intersection point of two lines -/
noncomputable def lineIntersection (l1 l2 : Line) : Point :=
  { x := (l1.b * l2.c - l2.b * l1.c) / (l1.a * l2.b - l2.a * l1.b),
    y := (l2.a * l1.c - l1.a * l2.c) / (l1.a * l2.b - l2.a * l1.b) }

/-- Theorem: Locus of point G forms a parabola -/
theorem locus_of_G_is_parabola (abc : RightTriangle) (d : Point) (ad ce ab : Line) :
  ∀ (e : Point), pointOnLine e ad →
  let f := lineIntersection ce ab
  let bc := Line.mk (abc.B.y - abc.C.y) (abc.C.x - abc.B.x) (abc.B.x * abc.C.y - abc.C.x * abc.B.y)
  let perpF := Line.mk bc.b (-bc.a) (-bc.b * f.x + bc.a * f.y)
  let be := Line.mk (e.y - abc.B.y) (abc.B.x - e.x) (e.x * abc.B.y - abc.B.x * e.y)
  let g := lineIntersection perpF be
  ∃ (a b : ℝ), g.y = (a / (b^2)) * (g.x - b)^2 := by
    sorry

end locus_of_G_is_parabola_l1604_160402


namespace system_solution_l1604_160468

theorem system_solution (m n : ℝ) 
  (eq1 : m * 3 + (-7) = 5)
  (eq2 : 2 * (7/2) - n * (-2) = 13)
  : ∃ (x y : ℝ), m * x + y = 5 ∧ 2 * x - n * y = 13 ∧ x = 2 ∧ y = -3 := by
  sorry

end system_solution_l1604_160468


namespace subtracted_value_l1604_160423

theorem subtracted_value (x y : ℝ) : 
  (x - 5) / 7 = 7 → 
  (x - y) / 10 = 2 → 
  y = 34 := by
sorry

end subtracted_value_l1604_160423


namespace lucy_has_19_snowballs_l1604_160407

-- Define the number of snowballs Charlie and Lucy have
def charlie_snowballs : ℕ := 50
def lucy_snowballs : ℕ := charlie_snowballs - 31

-- Theorem statement
theorem lucy_has_19_snowballs : lucy_snowballs = 19 := by
  sorry

end lucy_has_19_snowballs_l1604_160407


namespace k_range_l1604_160408

theorem k_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) → 
  -1 < k ∧ k ≤ 0 := by
sorry

end k_range_l1604_160408


namespace rectangle_ratio_l1604_160462

/-- A rectangle with a circle passing through two vertices and touching one side. -/
structure RectangleWithCircle where
  /-- Length of the longer side of the rectangle -/
  x : ℝ
  /-- Length of the shorter side of the rectangle -/
  y : ℝ
  /-- Radius of the circle -/
  R : ℝ
  /-- The perimeter of the rectangle is 4 times the radius of the circle -/
  h_perimeter : x + y = 2 * R
  /-- The circle passes through two vertices and touches one side -/
  h_circle_touch : y = R + Real.sqrt (R^2 - (x/2)^2)
  /-- The sides are positive -/
  h_positive : x > 0 ∧ y > 0 ∧ R > 0

/-- The ratio of the sides of the rectangle is 4:1 -/
theorem rectangle_ratio (rect : RectangleWithCircle) : rect.x / rect.y = 4 := by
  sorry

end rectangle_ratio_l1604_160462


namespace largest_valid_number_l1604_160451

def is_valid (n : ℕ) : Prop :=
  (n ≥ 1000000 ∧ n ≤ 9999999) ∧
  ∀ i : ℕ, i ∈ [0, 1, 2, 3, 4] →
    (((n / 10^i) % 1000) % 11 = 0 ∨ ((n / 10^i) % 1000) % 13 = 0)

theorem largest_valid_number :
  is_valid 9884737 ∧ ∀ m : ℕ, is_valid m → m ≤ 9884737 :=
sorry

end largest_valid_number_l1604_160451


namespace victorias_initial_money_l1604_160486

/-- Theorem: Victoria's Initial Money --/
theorem victorias_initial_money (rice_price wheat_price soda_price : ℕ)
  (rice_quantity wheat_quantity : ℕ) (remaining_balance : ℕ) :
  rice_price = 20 →
  wheat_price = 25 →
  soda_price = 150 →
  rice_quantity = 2 →
  wheat_quantity = 3 →
  remaining_balance = 235 →
  rice_quantity * rice_price + wheat_quantity * wheat_price + soda_price + remaining_balance = 500 :=
by
  sorry

#check victorias_initial_money

end victorias_initial_money_l1604_160486


namespace sqrt_pattern_l1604_160435

theorem sqrt_pattern (n : ℕ) (h : n ≥ 2) :
  Real.sqrt (n - n / (n^2 + 1)) = n * Real.sqrt (n / (n^2 + 1)) := by
  sorry

end sqrt_pattern_l1604_160435


namespace other_number_proof_l1604_160436

theorem other_number_proof (A B : ℕ+) (hcf lcm : ℕ+) : 
  hcf = Nat.gcd A B →
  lcm = Nat.lcm A B →
  hcf = 12 →
  lcm = 396 →
  A = 36 →
  B = 132 := by
sorry

end other_number_proof_l1604_160436


namespace prime_pairs_satisfying_equation_l1604_160483

theorem prime_pairs_satisfying_equation : 
  ∀ x y : ℕ, 
    Prime x → Prime y → 
    (x^2 - y^2 = x * y^2 - 19) ↔ 
    ((x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7)) :=
by sorry

end prime_pairs_satisfying_equation_l1604_160483


namespace amys_haircut_l1604_160455

/-- Amy's haircut problem -/
theorem amys_haircut (initial_length : ℝ) (final_length : ℝ) (cut_length : ℝ)
  (h1 : initial_length = 11)
  (h2 : final_length = 7)
  (h3 : cut_length = initial_length - final_length) :
  cut_length = 4 := by sorry

end amys_haircut_l1604_160455


namespace line_perpendicular_planes_parallel_l1604_160438

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicularToPlane : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallelPlanes : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicularLines : Line → Line → Prop)

-- Define the "contained in" relation for a line in a plane
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_planes_parallel
  (l m : Line) (α β : Plane)
  (h1 : perpendicularToPlane l α)
  (h2 : containedIn m β) :
  (∀ x y, parallelPlanes x y → perpendicularLines l m) ∧
  ∃ x y, perpendicularLines x y ∧ ¬parallelPlanes α β :=
sorry

end line_perpendicular_planes_parallel_l1604_160438


namespace star_arrangement_exists_l1604_160452

/-- A type representing a star-like configuration with 11 positions --/
structure StarConfiguration :=
  (positions : Fin 11 → ℕ)

/-- The sum of numbers from 1 to 11 --/
def sum_1_to_11 : ℕ := (11 * 12) / 2

/-- The segments of the star configuration --/
def segments : List (Fin 11 × Fin 11 × Fin 11) := sorry

/-- The condition that all numbers from 1 to 11 are used exactly once --/
def valid_arrangement (config : StarConfiguration) : Prop :=
  (∀ n : Fin 11, ∃ p : Fin 11, config.positions p = n.val + 1) ∧
  (∀ p q : Fin 11, p ≠ q → config.positions p ≠ config.positions q)

/-- The condition that the sum of each segment is 18 --/
def segment_sum_18 (config : StarConfiguration) : Prop :=
  ∀ seg ∈ segments, 
    config.positions seg.1 + config.positions seg.2.1 + config.positions seg.2.2 = 18

/-- The main theorem: there exists a valid arrangement with segment sum 18 --/
theorem star_arrangement_exists : 
  ∃ (config : StarConfiguration), valid_arrangement config ∧ segment_sum_18 config := by
  sorry

end star_arrangement_exists_l1604_160452


namespace inequality_system_solution_set_l1604_160412

theorem inequality_system_solution_set :
  let S : Set ℝ := {x | 2 * x - 4 ≤ 0 ∧ -x + 1 < 0}
  S = {x | 1 < x ∧ x ≤ 2} := by
  sorry

end inequality_system_solution_set_l1604_160412


namespace sequence_a_formula_sequence_a_first_term_sequence_a_second_term_sequence_a_third_term_sequence_a_fourth_term_l1604_160472

def sequence_a (n : ℕ) : ℚ := (18 * n - 9) / (7 * (10^n - 1))

theorem sequence_a_formula (n : ℕ) : 
  sequence_a n = (18 * n - 9) / (7 * (10^n - 1)) :=
by sorry

theorem sequence_a_first_term : sequence_a 1 = 1 / 7 :=
by sorry

theorem sequence_a_second_term : sequence_a 2 = 3 / 77 :=
by sorry

theorem sequence_a_third_term : sequence_a 3 = 5 / 777 :=
by sorry

theorem sequence_a_fourth_term : sequence_a 4 = 7 / 7777 :=
by sorry

end sequence_a_formula_sequence_a_first_term_sequence_a_second_term_sequence_a_third_term_sequence_a_fourth_term_l1604_160472


namespace system_solution_unique_l1604_160480

theorem system_solution_unique : 
  ∃! (x y : ℝ), x + y = 2 ∧ x + 2*y = 3 :=
by
  -- The proof goes here
  sorry

end system_solution_unique_l1604_160480


namespace pastry_chef_eggs_l1604_160409

theorem pastry_chef_eggs :
  ∃ n : ℕ,
    n > 0 ∧
    n % 43 = 0 ∧
    n % 2 = 1 ∧
    n % 3 = 1 ∧
    n % 4 = 1 ∧
    n % 5 = 1 ∧
    n % 6 = 1 ∧
    n / 43 < 9 ∧
    (∀ m : ℕ, m > 0 ∧ m < n →
      ¬(m % 43 = 0 ∧
        m % 2 = 1 ∧
        m % 3 = 1 ∧
        m % 4 = 1 ∧
        m % 5 = 1 ∧
        m % 6 = 1 ∧
        m / 43 < 9)) ∧
    n = 301 := by
  sorry

end pastry_chef_eggs_l1604_160409


namespace least_multiple_first_ten_gt_1000_l1604_160470

theorem least_multiple_first_ten_gt_1000 : ∃ n : ℕ,
  n > 1000 ∧
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧
  (∀ m : ℕ, m > 1000 ∧ (∀ k : ℕ, k ≤ 10 → k > 0 → m % k = 0) → m ≥ n) ∧
  n = 2520 := by
sorry

end least_multiple_first_ten_gt_1000_l1604_160470


namespace sum_of_units_digits_equals_zero_l1604_160419

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the problem
theorem sum_of_units_digits_equals_zero :
  (unitsDigit (17 * 34) + unitsDigit (19 * 28)) % 10 = 0 := by
  sorry

end sum_of_units_digits_equals_zero_l1604_160419


namespace school_raffle_earnings_l1604_160424

/-- The amount of money Zoe's school made from selling raffle tickets -/
def total_money_made (cost_per_ticket : ℕ) (num_tickets_sold : ℕ) : ℕ :=
  cost_per_ticket * num_tickets_sold

/-- Theorem stating that Zoe's school made 620 dollars from selling raffle tickets -/
theorem school_raffle_earnings :
  total_money_made 4 155 = 620 := by
  sorry

end school_raffle_earnings_l1604_160424


namespace remaining_numbers_are_even_l1604_160406

def last_digit (n : ℕ) : ℕ := n % 10
def second_last_digit (n : ℕ) : ℕ := (n / 10) % 10

def is_removed (n : ℕ) : Prop :=
  (last_digit n % 2 = 1 ∧ second_last_digit n % 2 = 0) ∨
  (last_digit n % 2 = 1 ∧ last_digit n % 3 ≠ 0) ∨
  (second_last_digit n % 2 = 1 ∧ n % 3 = 0)

theorem remaining_numbers_are_even (n : ℕ) :
  ¬(is_removed n) → Even n :=
by sorry

end remaining_numbers_are_even_l1604_160406


namespace circle_circumference_l1604_160415

/-- The circumference of a circle with diameter 2 yards is equal to π * 2 yards. -/
theorem circle_circumference (diameter : ℝ) (h : diameter = 2) :
  (diameter * π) = 2 * π := by sorry

end circle_circumference_l1604_160415


namespace art_class_price_l1604_160400

/-- Represents the price of Claudia's one-hour art class -/
def class_price : ℝ := 10

/-- Number of kids attending Saturday's class -/
def saturday_attendance : ℕ := 20

/-- Number of kids attending Sunday's class -/
def sunday_attendance : ℕ := saturday_attendance / 2

/-- Total earnings for both days -/
def total_earnings : ℝ := 300

theorem art_class_price :
  class_price * (saturday_attendance + sunday_attendance) = total_earnings :=
sorry

end art_class_price_l1604_160400


namespace patrol_impossibility_l1604_160465

/-- Represents the number of people in the group -/
def n : ℕ := 100

/-- Represents the number of people on duty each evening -/
def k : ℕ := 3

/-- Represents the total number of possible pairs of people -/
def totalPairs : ℕ := n.choose 2

/-- Represents the number of pairs formed each evening -/
def pairsPerEvening : ℕ := k.choose 2

theorem patrol_impossibility : ¬ ∃ (m : ℕ), m * pairsPerEvening = totalPairs ∧ 
  ∃ (f : Fin n → Fin m → Bool), 
    (∀ i j, i ≠ j → (∃! t, f i t ∧ f j t)) ∧
    (∀ t, ∃! (s : Fin k → Fin n), (∀ i, f (s i) t)) :=
sorry

end patrol_impossibility_l1604_160465


namespace line_through_points_2m_plus_3b_l1604_160466

/-- Given a line passing through the points (-1, 1/2) and (2, -3/2), 
    prove that 2m+3b = -11/6 when the line is expressed as y = mx + b -/
theorem line_through_points_2m_plus_3b (m b : ℚ) : 
  (1/2 : ℚ) = m * (-1) + b →
  (-3/2 : ℚ) = m * 2 + b →
  2 * m + 3 * b = -11/6 := by
  sorry

end line_through_points_2m_plus_3b_l1604_160466


namespace expansion_coefficient_l1604_160469

theorem expansion_coefficient (m : ℤ) : 
  (Nat.choose 6 3 : ℤ) * m^3 = -160 → m = -2 := by
  sorry

end expansion_coefficient_l1604_160469


namespace fifteen_factorial_base_eight_zeroes_l1604_160432

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- Theorem: 15! ends with 3 zeroes when written in base 8 -/
theorem fifteen_factorial_base_eight_zeroes :
  trailingZeroes 15 8 = 3 := by
  sorry

end fifteen_factorial_base_eight_zeroes_l1604_160432


namespace chemistry_class_section_size_l1604_160429

theorem chemistry_class_section_size :
  let section1_size : ℕ := 65
  let section2_size : ℕ := 35
  let section4_size : ℕ := 42
  let section1_mean : ℚ := 50
  let section2_mean : ℚ := 60
  let section3_mean : ℚ := 55
  let section4_mean : ℚ := 45
  let overall_mean : ℚ := 5195 / 100

  ∃ (section3_size : ℕ),
    (section1_size * section1_mean + section2_size * section2_mean + 
     section3_size * section3_mean + section4_size * section4_mean) / 
    (section1_size + section2_size + section3_size + section4_size : ℚ) = overall_mean ∧
    section3_size = 45
  := by sorry

end chemistry_class_section_size_l1604_160429


namespace ellipse_foci_distance_l1604_160422

theorem ellipse_foci_distance (a b : ℝ) (h1 : a = 6) (h2 : b = 2) :
  2 * Real.sqrt (a^2 - b^2) = 8 * Real.sqrt 2 := by
  sorry

end ellipse_foci_distance_l1604_160422


namespace cos_330_degrees_l1604_160497

theorem cos_330_degrees : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l1604_160497


namespace males_in_band_not_in_orchestra_l1604_160448

/-- Given the information about band and orchestra membership, prove that the number of males in the band who are not in the orchestra is 10. -/
theorem males_in_band_not_in_orchestra : 
  ∀ (female_band male_band female_orch male_orch female_both total_students : ℕ),
    female_band = 100 →
    male_band = 80 →
    female_orch = 80 →
    male_orch = 100 →
    female_both = 60 →
    total_students = 230 →
    ∃ (male_both : ℕ),
      female_band + female_orch - female_both + male_band + male_orch - male_both = total_students ∧
      male_band - male_both = 10 :=
by sorry

end males_in_band_not_in_orchestra_l1604_160448


namespace orange_eaters_ratio_l1604_160454

/-- Represents a family gathering with a specific number of people and orange eaters. -/
structure FamilyGathering where
  total_people : ℕ
  orange_eaters : ℕ
  h_orange_eaters : orange_eaters = total_people - 10

/-- The ratio of orange eaters to total people in a family gathering is 1:2. -/
theorem orange_eaters_ratio (gathering : FamilyGathering) 
    (h_total : gathering.total_people = 20) : 
    (gathering.orange_eaters : ℚ) / gathering.total_people = 1 / 2 := by
  sorry


end orange_eaters_ratio_l1604_160454


namespace largest_divisor_of_9670_l1604_160413

theorem largest_divisor_of_9670 : ∃ (d : ℕ), d > 0 ∧ d ∣ (9671 - 1) ∧ ∀ (x : ℕ), x > 0 → x ∣ (9671 - 1) → x ≤ d := by
  sorry

end largest_divisor_of_9670_l1604_160413


namespace fraction_simplification_l1604_160476

theorem fraction_simplification (x : ℝ) (h : x = 5) :
  (x^6 - 25*x^3 + 144) / (x^3 - 12) = 114 := by
  sorry

end fraction_simplification_l1604_160476


namespace square_area_relation_l1604_160417

theorem square_area_relation (a b : ℝ) : 
  let diagonal_I : ℝ := 2*a + 3*b
  let area_I : ℝ := (diagonal_I^2) / 2
  let area_II : ℝ := area_I^3
  area_II = (diagonal_I^6) / 8 := by
sorry

end square_area_relation_l1604_160417


namespace lcm_of_54_96_120_150_l1604_160442

theorem lcm_of_54_96_120_150 : Nat.lcm 54 (Nat.lcm 96 (Nat.lcm 120 150)) = 21600 := by
  sorry

end lcm_of_54_96_120_150_l1604_160442


namespace sum_of_coefficients_l1604_160418

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 1)^5 = a₅*(x + 1)^5 + a₄*(x + 1)^4 + a₃*(x + 1)^3 + a₂*(x + 1)^2 + a₁*(x + 1) + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end sum_of_coefficients_l1604_160418


namespace parabola_equation_l1604_160493

def is_valid_parabola (p : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ),
    (2 * x₁ + 1)^2 = 2 * p * x₁ ∧
    (2 * x₂ + 1)^2 = 2 * p * x₂ ∧
    (x₁ - x₂)^2 * 5 = 15

theorem parabola_equation :
  ∀ p : ℝ, is_valid_parabola p → (p = -2 ∨ p = 6) := by sorry

end parabola_equation_l1604_160493


namespace gcd_digits_bound_l1604_160456

theorem gcd_digits_bound (a b : ℕ) : 
  (1000000 ≤ a ∧ a < 10000000) →
  (1000000 ≤ b ∧ b < 10000000) →
  (100000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000000) →
  Nat.gcd a b < 1000 :=
by sorry

end gcd_digits_bound_l1604_160456


namespace cards_in_same_envelope_probability_l1604_160450

/-- The number of cards -/
def num_cards : ℕ := 6

/-- The number of envelopes -/
def num_envelopes : ℕ := 3

/-- The number of cards per envelope -/
def cards_per_envelope : ℕ := 2

/-- The set of all possible distributions of cards into envelopes -/
def all_distributions : Finset (Fin num_cards → Fin num_envelopes) :=
  sorry

/-- The set of distributions where cards 1 and 2 are in the same envelope -/
def favorable_distributions : Finset (Fin num_cards → Fin num_envelopes) :=
  sorry

/-- The probability of cards 1 and 2 being in the same envelope -/
def prob_same_envelope : ℚ :=
  (favorable_distributions.card : ℚ) / (all_distributions.card : ℚ)

theorem cards_in_same_envelope_probability :
  prob_same_envelope = 1 / 5 :=
sorry

end cards_in_same_envelope_probability_l1604_160450


namespace quadratic_rewrite_l1604_160498

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x : ℝ, 4 * x^2 - 16 * x + 2 = (d * x + e)^2 + f) → d * e = -8 := by
  sorry

end quadratic_rewrite_l1604_160498


namespace star_polygon_points_l1604_160420

/-- A regular star polygon with n points, where each point has two types of angles -/
structure StarPolygon where
  n : ℕ
  angle_A : ℝ
  angle_B : ℝ
  h1 : angle_A = angle_B - 15
  h2 : 0 < n

/-- The sum of all exterior angles in a polygon is 360° -/
axiom sum_of_exterior_angles : ∀ (p : StarPolygon), p.n * (p.angle_B - p.angle_A) = 360

/-- The number of points in the star polygon is 24 -/
theorem star_polygon_points (p : StarPolygon) : p.n = 24 := by
  sorry

end star_polygon_points_l1604_160420


namespace optimal_profit_l1604_160444

/-- Represents the profit optimization problem for a shopping mall --/
structure ShoppingMall where
  total_boxes : ℕ
  profit_A : ℝ
  profit_B : ℝ
  profit_diff : ℝ
  price_change : ℝ
  box_change : ℝ

/-- Calculates the optimal price reduction and maximum profit --/
def optimize_profit (mall : ShoppingMall) : ℝ × ℝ :=
  sorry

/-- Theorem stating the optimal price reduction and maximum profit --/
theorem optimal_profit (mall : ShoppingMall) 
  (h1 : mall.total_boxes = 600)
  (h2 : mall.profit_A = 40000)
  (h3 : mall.profit_B = 160000)
  (h4 : mall.profit_diff = 200)
  (h5 : mall.price_change = 5)
  (h6 : mall.box_change = 2) :
  optimize_profit mall = (75, 204500) :=
sorry

end optimal_profit_l1604_160444


namespace sam_hunts_seven_l1604_160489

/-- The number of animals hunted by Sam, Rob, Mark, and Peter in a day -/
def total_animals : ℕ := 21

/-- Sam's hunt count -/
def sam_hunt : ℕ := 7

/-- Rob's hunt count in terms of Sam's -/
def rob_hunt (s : ℕ) : ℚ := s / 2

/-- Mark's hunt count in terms of Sam's -/
def mark_hunt (s : ℕ) : ℚ := (1 / 3) * (s + rob_hunt s)

/-- Peter's hunt count in terms of Sam's -/
def peter_hunt (s : ℕ) : ℚ := 3 * mark_hunt s

/-- Theorem stating that Sam hunts 7 animals given the conditions -/
theorem sam_hunts_seven :
  sam_hunt + rob_hunt sam_hunt + mark_hunt sam_hunt + peter_hunt sam_hunt = total_animals := by
  sorry

#eval sam_hunt

end sam_hunts_seven_l1604_160489


namespace imaginary_product_implies_a_value_l1604_160414

theorem imaginary_product_implies_a_value (a : ℝ) : 
  (∃ b : ℝ, (a - Complex.I) * (3 - 2 * Complex.I) = b * Complex.I) → a = 2/3 :=
by sorry

end imaginary_product_implies_a_value_l1604_160414


namespace inverse_sum_simplification_l1604_160430

theorem inverse_sum_simplification (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ - y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (y * z - x * z + x * y) := by
  sorry

end inverse_sum_simplification_l1604_160430


namespace max_integer_a_for_real_roots_l1604_160441

theorem max_integer_a_for_real_roots : 
  ∀ a : ℤ, 
  (a ≠ 1) → 
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 3 = 0) → 
  a ≤ 0 ∧ 
  ∀ b : ℤ, b > 0 → ¬(∃ x : ℝ, (b - 1) * x^2 - 2 * x + 3 = 0) :=
by sorry

end max_integer_a_for_real_roots_l1604_160441


namespace equation_solutions_l1604_160426

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧
    x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 2/3 ∧
    3*x₁*(x₁-1) = 2*x₁-2 ∧ 3*x₂*(x₂-1) = 2*x₂-2) :=
by
  sorry

end equation_solutions_l1604_160426


namespace f_negation_l1604_160434

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 2

-- State the theorem
theorem f_negation (a b : ℝ) :
  f a b 2011 = 10 → f a b (-2011) = -14 := by
  sorry

end f_negation_l1604_160434


namespace total_ladybugs_count_l1604_160481

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := ladybugs_with_spots + ladybugs_without_spots

theorem total_ladybugs_count : total_ladybugs = 67082 := by
  sorry

end total_ladybugs_count_l1604_160481


namespace sum_first_23_equals_11_l1604_160490

def repeatingSequence : List Int := [4, -3, 2, -1, 0]

def sumFirstN (n : Nat) : Int :=
  let fullCycles := n / repeatingSequence.length
  let remainder := n % repeatingSequence.length
  fullCycles * repeatingSequence.sum +
    (repeatingSequence.take remainder).sum

theorem sum_first_23_equals_11 : sumFirstN 23 = 11 := by
  sorry

end sum_first_23_equals_11_l1604_160490


namespace goat_difference_l1604_160439

-- Define the number of goats for each person
def adam_goats : ℕ := 7
def ahmed_goats : ℕ := 13

-- Define Andrew's goats in terms of Adam's
def andrew_goats : ℕ := 2 * adam_goats + 5

-- Theorem statement
theorem goat_difference : andrew_goats - ahmed_goats = 6 := by
  sorry

end goat_difference_l1604_160439


namespace divisibility_problem_l1604_160457

theorem divisibility_problem (N : ℕ) (h1 : N % 44 = 0) (h2 : N % 30 = 18) : N / 44 = 3 := by
  sorry

end divisibility_problem_l1604_160457


namespace max_intersections_four_circles_l1604_160440

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Predicate to check if a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop :=
  sorry

/-- The number of intersection points between a line and a circle -/
def numIntersections (l : Line) (c : Circle) : ℕ :=
  sorry

/-- Predicate to check if four circles are coplanar -/
def coplanar (c1 c2 c3 c4 : Circle) : Prop :=
  sorry

/-- Theorem: The maximum number of intersection points between a line and four coplanar circles is 8 -/
theorem max_intersections_four_circles (c1 c2 c3 c4 : Circle) (l : Line) :
  coplanar c1 c2 c3 c4 →
  intersects l c1 →
  intersects l c2 →
  intersects l c3 →
  intersects l c4 →
  numIntersections l c1 + numIntersections l c2 + numIntersections l c3 + numIntersections l c4 ≤ 8 :=
sorry

end max_intersections_four_circles_l1604_160440


namespace total_turnips_l1604_160475

def keith_turnips : ℕ := 6
def alyssa_turnips : ℕ := 9

theorem total_turnips : keith_turnips + alyssa_turnips = 15 := by
  sorry

end total_turnips_l1604_160475


namespace three_digit_number_problem_l1604_160411

theorem three_digit_number_problem : ∃ (a b c : ℕ), 
  (8 * a + 5 * b + c = 100) ∧ 
  (a + b + c = 20) ∧ 
  (a < 10) ∧ (b < 10) ∧ (c < 10) ∧
  (a * 100 + b * 10 + c = 866) := by
  sorry

end three_digit_number_problem_l1604_160411


namespace decimal_repetend_of_five_thirteenth_l1604_160496

/-- The decimal representation of 5/13 has a 6-digit repetend of 384615 -/
theorem decimal_repetend_of_five_thirteenth : ∃ (n : ℕ), 
  (5 : ℚ) / 13 = (384615 : ℚ) / 999999 + (n : ℚ) / 999999 := by
  sorry

end decimal_repetend_of_five_thirteenth_l1604_160496


namespace means_inequality_l1604_160471

theorem means_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h_max : max b c ≥ (a + b) / 2) : 
  Real.sqrt ((b^2 + c^2) / 2) > (a + b) / 2 ∧ 
  (a + b) / 2 > Real.sqrt (a * b) ∧ 
  Real.sqrt (a * b) > 2 * a * b / (a + b) := by
  sorry

end means_inequality_l1604_160471


namespace unique_divisor_function_exists_l1604_160425

open Nat

/-- The divisor function τ(n) counts the number of positive divisors of n. -/
noncomputable def tau (n : ℕ) : ℕ := (divisors n).card

/-- 
Given a finite set of natural numbers, there exists a number x such that 
the divisor function τ applied to the product of x and any element of the set 
yields a unique result for each element of the set.
-/
theorem unique_divisor_function_exists (S : Finset ℕ) : 
  ∃ x : ℕ, ∀ s₁ s₂ : ℕ, s₁ ∈ S → s₂ ∈ S → s₁ ≠ s₂ → tau (s₁ * x) ≠ tau (s₂ * x) := by
  sorry

end unique_divisor_function_exists_l1604_160425
