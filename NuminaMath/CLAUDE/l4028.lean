import Mathlib

namespace arithmetic_mean_problem_l4028_402808

theorem arithmetic_mean_problem (a₁ a₂ a₃ a₄ a₅ a₆ A : ℝ) 
  (h1 : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = A)
  (h2 : (a₁ + a₂ + a₃ + a₄) / 4 = A + 10)
  (h3 : (a₃ + a₄ + a₅ + a₆) / 4 = A - 7) :
  (a₁ + a₂ + a₅ + a₆) / 4 = A - 3 := by
  sorry

end arithmetic_mean_problem_l4028_402808


namespace smallest_prime_factor_of_2343_l4028_402847

theorem smallest_prime_factor_of_2343 : 
  Nat.minFac 2343 = 3 := by
sorry

end smallest_prime_factor_of_2343_l4028_402847


namespace total_harvest_l4028_402887

/-- The number of sacks of oranges harvested per day -/
def daily_harvest : ℕ := 83

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- Theorem stating the total number of sacks harvested after 6 days -/
theorem total_harvest : daily_harvest * harvest_days = 498 := by
  sorry

end total_harvest_l4028_402887


namespace quadratic_root_problem_l4028_402893

theorem quadratic_root_problem (m : ℝ) :
  (1 : ℝ) ^ 2 - 4 * (1 : ℝ) + m + 1 = 0 →
  m = 2 ∧ ∃ (x : ℝ), x ≠ 1 ∧ x ^ 2 - 4 * x + m + 1 = 0 ∧ x = 3 :=
by sorry

end quadratic_root_problem_l4028_402893


namespace unique_peg_arrangement_l4028_402828

/-- Represents a color of a peg -/
inductive PegColor
  | Yellow
  | Red
  | Green
  | Blue
  | Orange

/-- Represents a position on the triangular peg board -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the triangular peg board -/
def Board := Position → Option PegColor

/-- Checks if a given board arrangement is valid -/
def is_valid_arrangement (board : Board) : Prop :=
  (∀ r c, board ⟨r, c⟩ = some PegColor.Yellow → r < 6 ∧ c < 6) ∧
  (∀ r c, board ⟨r, c⟩ = some PegColor.Red → r < 5 ∧ c < 6) ∧
  (∀ r c, board ⟨r, c⟩ = some PegColor.Green → r < 4 ∧ c < 6) ∧
  (∀ r c, board ⟨r, c⟩ = some PegColor.Blue → r < 3 ∧ c < 6) ∧
  (∀ r c, board ⟨r, c⟩ = some PegColor.Orange → r < 2 ∧ c < 6) ∧
  (∀ r, ∃! c, board ⟨r, c⟩ = some PegColor.Yellow) ∧
  (∀ r, r < 5 → ∃! c, board ⟨r, c⟩ = some PegColor.Red) ∧
  (∀ r, r < 4 → ∃! c, board ⟨r, c⟩ = some PegColor.Green) ∧
  (∀ r, r < 3 → ∃! c, board ⟨r, c⟩ = some PegColor.Blue) ∧
  (∀ r, r < 2 → ∃! c, board ⟨r, c⟩ = some PegColor.Orange) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Yellow) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Red) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Green) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Blue) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Orange)

theorem unique_peg_arrangement :
  ∃! board : Board, is_valid_arrangement board :=
sorry

end unique_peg_arrangement_l4028_402828


namespace first_sequence_30th_term_l4028_402848

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- The 30th term of the first arithmetic sequence is 178 -/
theorem first_sequence_30th_term :
  arithmeticSequenceTerm 4 6 30 = 178 := by
  sorry

end first_sequence_30th_term_l4028_402848


namespace cliffs_rock_collection_l4028_402883

/-- The number of rocks in Cliff's collection -/
def total_rocks (igneous sedimentary metamorphic comet : ℕ) : ℕ :=
  igneous + sedimentary + metamorphic + comet

theorem cliffs_rock_collection :
  ∀ (igneous sedimentary metamorphic comet : ℕ),
    igneous = sedimentary / 2 →
    metamorphic = igneous / 3 →
    comet = 2 * metamorphic →
    igneous / 4 = 15 →
    comet / 2 = 20 →
    total_rocks igneous sedimentary metamorphic comet = 240 := by
  sorry

end cliffs_rock_collection_l4028_402883


namespace vector_parallel_implies_x_value_l4028_402843

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is parallel to c, then the x-coordinate of a is -11/3 -/
theorem vector_parallel_implies_x_value 
  (a b c : ℝ × ℝ) 
  (hb : b = (1, 2)) 
  (hc : c = (-1, 3)) 
  (ha : a.2 = 1) 
  (h_parallel : ∃ (k : ℝ), (a + 2 • b) = k • c) : 
  a.1 = -11/3 := by sorry

end vector_parallel_implies_x_value_l4028_402843


namespace heather_oranges_l4028_402856

def oranges_problem (initial : ℕ) (russell_takes : ℕ) (samantha_takes : ℕ) : Prop :=
  initial - russell_takes - samantha_takes = 13

theorem heather_oranges :
  oranges_problem 60 35 12 := by
  sorry

end heather_oranges_l4028_402856


namespace smallest_n_perfect_powers_l4028_402827

theorem smallest_n_perfect_powers : ∃ (n : ℕ),
  (n = 1944) ∧
  (∃ (m : ℕ), 2 * n = m^4) ∧
  (∃ (l : ℕ), 3 * n = l^6) ∧
  (∀ (k : ℕ), k < n →
    (∃ (p : ℕ), 2 * k = p^4) →
    (∃ (q : ℕ), 3 * k = q^6) →
    False) :=
by sorry

end smallest_n_perfect_powers_l4028_402827


namespace solution_set_of_inequality_l4028_402871

open Set
open Function

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the derivative of f
noncomputable def f' : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_of_inequality 
  (hf_domain : ∀ x, x ∈ (Set.Ioi 0) → DifferentiableAt ℝ f x)
  (hf'_def : ∀ x, x ∈ (Set.Ioi 0) → HasDerivAt f (f' x) x)
  (hf'_condition : ∀ x, x ∈ (Set.Ioi 0) → x * f' x > f x) :
  {x : ℝ | (x - 1) * f (x + 1) > f (x^2 - 1)} = Ioo 1 2 :=
sorry

end solution_set_of_inequality_l4028_402871


namespace determine_F_l4028_402844

def first_number (D E : ℕ) : ℕ := 9000000 + 600000 + 100000 * D + 10000 + 1000 * E + 800 + 2

def second_number (D E F : ℕ) : ℕ := 5000000 + 400000 + 100000 * E + 10000 * D + 2000 + 100 + 10 * F

theorem determine_F :
  ∀ D E F : ℕ,
  D < 10 → E < 10 → F < 10 →
  (first_number D E) % 3 = 0 →
  (second_number D E F) % 3 = 0 →
  F = 2 := by
sorry

end determine_F_l4028_402844


namespace first_group_men_count_l4028_402878

/-- Represents the amount of work that can be done by one person in one day -/
structure WorkRate where
  men : ℝ
  boys : ℝ

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  boys : ℕ

/-- Represents a work scenario -/
structure WorkScenario where
  group : WorkGroup
  days : ℕ

theorem first_group_men_count (rate : WorkRate) 
  (scenario1 : WorkScenario)
  (scenario2 : WorkScenario)
  (scenario3 : WorkScenario) :
  scenario1.group.men = 6 :=
by
  sorry

#check first_group_men_count

end first_group_men_count_l4028_402878


namespace min_value_zero_at_k_eq_two_l4028_402814

/-- The quadratic function f(x, y) depending on parameter k -/
def f (k : ℝ) (x y : ℝ) : ℝ :=
  4 * x^2 - 6 * k * x * y + (3 * k^2 + 2) * y^2 - 4 * x - 4 * y + 6

/-- Theorem stating that k = 2 is the unique value for which the minimum of f is 0 -/
theorem min_value_zero_at_k_eq_two :
  ∃! k : ℝ, (∀ x y : ℝ, f k x y ≥ 0) ∧ (∃ x y : ℝ, f k x y = 0) :=
by
  sorry

end min_value_zero_at_k_eq_two_l4028_402814


namespace abs_g_zero_equals_70_l4028_402811

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- Condition that g is a third-degree polynomial with specific absolute values -/
def SatisfiesCondition (g : ThirdDegreePolynomial) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d) ∧
  (|g 1| = 10) ∧ (|g 3| = 10) ∧ (|g 4| = 10) ∧
  (|g 6| = 10) ∧ (|g 8| = 10) ∧ (|g 9| = 10)

/-- Theorem: If g satisfies the condition, then |g(0)| = 70 -/
theorem abs_g_zero_equals_70 (g : ThirdDegreePolynomial) 
  (h : SatisfiesCondition g) : |g 0| = 70 := by
  sorry

end abs_g_zero_equals_70_l4028_402811


namespace outfit_combinations_l4028_402802

def num_shirts : ℕ := 8
def num_pants : ℕ := 5
def num_jacket_options : ℕ := 3

theorem outfit_combinations :
  num_shirts * num_pants * num_jacket_options = 120 :=
by sorry

end outfit_combinations_l4028_402802


namespace unique_real_root_l4028_402836

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - 10

-- Theorem statement
theorem unique_real_root : ∃! x : ℝ, f x = 0 := by
  sorry

end unique_real_root_l4028_402836


namespace karen_crayons_count_l4028_402846

/-- The number of crayons Cindy has -/
def cindy_crayons : ℕ := 504

/-- The number of additional crayons Karen has compared to Cindy -/
def karen_additional_crayons : ℕ := 135

/-- The number of crayons Karen has -/
def karen_crayons : ℕ := cindy_crayons + karen_additional_crayons

theorem karen_crayons_count : karen_crayons = 639 := by
  sorry

end karen_crayons_count_l4028_402846


namespace unique_quadratic_solution_l4028_402870

/-- The number of positive single-digit integers A for which x^2 - (2A + 1)x + 3A = 0 has positive integer solutions is 1. -/
theorem unique_quadratic_solution : 
  ∃! (A : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 
  ∃ (x : ℕ), x > 0 ∧ x^2 - (2 * A + 1) * x + 3 * A = 0 :=
by sorry

end unique_quadratic_solution_l4028_402870


namespace simplify_and_evaluate_l4028_402818

theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  (x^2 - 1) / (x + 2) / (1 - 1 / (x + 2)) = -4 := by
  sorry

end simplify_and_evaluate_l4028_402818


namespace factor_x_sixth_plus_64_l4028_402800

theorem factor_x_sixth_plus_64 (x : ℝ) : x^6 + 64 = (x^2 + 4) * (x^4 - 4*x^2 + 16) := by
  sorry

end factor_x_sixth_plus_64_l4028_402800


namespace partnership_gain_l4028_402892

/-- Represents the annual gain of a partnership given investments and durations -/
def annual_gain (x : ℝ) : ℝ :=
  let a_investment := x * 12
  let b_investment := 2 * x * 6
  let c_investment := 3 * x * 4
  let total_investment := a_investment + b_investment + c_investment
  let a_share := 6400
  3 * a_share

/-- Theorem stating that the annual gain of the partnership is 19200 -/
theorem partnership_gain : annual_gain x = 19200 :=
sorry

end partnership_gain_l4028_402892


namespace dark_king_game_winner_l4028_402829

/-- The dark king game on an n × m chessboard -/
def DarkKingGame (n m : ℕ) :=
  { board : Set (ℕ × ℕ) // board ⊆ (Finset.range n).product (Finset.range m) }

/-- A player in the dark king game -/
inductive Player
| First
| Second

/-- A winning strategy for a player in the dark king game -/
def WinningStrategy (n m : ℕ) (p : Player) :=
  ∃ (strategy : DarkKingGame n m → ℕ × ℕ),
    ∀ (game : DarkKingGame n m),
      (strategy game ∉ game.val) →
      (strategy game).1 < n ∧ (strategy game).2 < m

/-- The main theorem about the dark king game -/
theorem dark_king_game_winner (n m : ℕ) :
  (n % 2 = 0 ∨ m % 2 = 0) → WinningStrategy n m Player.First ∧
  (n % 2 = 1 ∧ m % 2 = 1) → WinningStrategy n m Player.Second :=
sorry

end dark_king_game_winner_l4028_402829


namespace sqrt_of_sqrt_16_over_81_l4028_402857

theorem sqrt_of_sqrt_16_over_81 : Real.sqrt (Real.sqrt (16 / 81)) = 2 / 3 := by
  sorry

end sqrt_of_sqrt_16_over_81_l4028_402857


namespace characterization_of_good_numbers_l4028_402895

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → (d + 1) ∣ (n + 1)

theorem characterization_of_good_numbers (n : ℕ) :
  is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1) :=
sorry

end characterization_of_good_numbers_l4028_402895


namespace geometric_sequence_extreme_points_l4028_402841

/-- Given a geometric sequence {a_n} where a_3 and a_7 are extreme points of f(x) = (1/3)x^3 + 4x^2 + 9x - 1, prove a_5 = -3 -/
theorem geometric_sequence_extreme_points (a : ℕ → ℝ) (h_geometric : ∀ n, a (n+1) / a n = a (n+2) / a (n+1)) :
  (∀ x, (x^2 + 8*x + 9) * (x - a 3) * (x - a 7) ≥ 0) →
  a 3 * a 7 = 9 →
  a 3 + a 7 = -8 →
  a 5 = -3 := by
sorry

end geometric_sequence_extreme_points_l4028_402841


namespace order_of_special_values_l4028_402849

/-- Given a = √(1.01), b = e^(0.01) / 1.01, and c = ln(1.01e), prove that b < a < c. -/
theorem order_of_special_values :
  let a : ℝ := Real.sqrt 1.01
  let b : ℝ := Real.exp 0.01 / 1.01
  let c : ℝ := Real.log (1.01 * Real.exp 1)
  b < a ∧ a < c := by sorry

end order_of_special_values_l4028_402849


namespace dressing_p_vinegar_percent_l4028_402831

/-- Represents a salad dressing with a specific percentage of vinegar -/
structure SaladDressing where
  vinegar_percent : ℝ
  oil_percent : ℝ
  vinegar_oil_sum : vinegar_percent + oil_percent = 100

/-- The percentage of dressing P in the new mixture -/
def p_mixture_percent : ℝ := 10

/-- The percentage of dressing Q in the new mixture -/
def q_mixture_percent : ℝ := 100 - p_mixture_percent

/-- Dressing Q contains 10% vinegar -/
def dressing_q : SaladDressing := ⟨10, 90, by norm_num⟩

/-- The percentage of vinegar in the new mixture -/
def new_mixture_vinegar_percent : ℝ := 12

/-- Theorem stating that dressing P contains 30% vinegar -/
theorem dressing_p_vinegar_percent :
  ∃ (dressing_p : SaladDressing),
    dressing_p.vinegar_percent = 30 ∧
    (p_mixture_percent / 100 * dressing_p.vinegar_percent +
     q_mixture_percent / 100 * dressing_q.vinegar_percent = new_mixture_vinegar_percent) :=
by sorry

end dressing_p_vinegar_percent_l4028_402831


namespace negation_equivalence_l4028_402873

theorem negation_equivalence (m : ℤ) : 
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by sorry

end negation_equivalence_l4028_402873


namespace power_equality_l4028_402879

theorem power_equality (m : ℕ) : 9^4 = 3^m → m = 8 := by
  sorry

end power_equality_l4028_402879


namespace b_work_rate_l4028_402832

/-- Given work rates for individuals and groups, prove B's work rate -/
theorem b_work_rate 
  (a_rate : ℚ)
  (b_rate : ℚ)
  (c_rate : ℚ)
  (d_rate : ℚ)
  (h1 : a_rate = 1/4)
  (h2 : b_rate + c_rate = 1/2)
  (h3 : a_rate + c_rate = 1/2)
  (h4 : d_rate = 1/8)
  (h5 : a_rate + b_rate + d_rate = 1/(8/5)) :
  b_rate = 1/4 := by
sorry

end b_work_rate_l4028_402832


namespace number_of_possible_scores_l4028_402880

-- Define the scoring system
def problem_scores : List Nat := [1, 2, 3, 4]
def time_bonuses : List Nat := [1, 2, 3, 4]
def all_correct_bonus : Nat := 20

-- Function to calculate all possible scores
def calculate_scores : List Nat :=
  let base_scores := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let multiplied_scores := 
    List.join (base_scores.map (λ s => time_bonuses.map (λ t => s * t)))
  let all_correct_scores := 
    time_bonuses.map (λ t => 10 * t + all_correct_bonus)
  List.eraseDups (multiplied_scores ++ all_correct_scores)

-- Theorem statement
theorem number_of_possible_scores : 
  calculate_scores.length = 25 := by sorry

end number_of_possible_scores_l4028_402880


namespace expression_equals_sqrt_two_l4028_402807

theorem expression_equals_sqrt_two : (-1)^2 + |-Real.sqrt 2| + (Real.pi - 3)^0 - Real.sqrt 4 = Real.sqrt 2 := by
  sorry

end expression_equals_sqrt_two_l4028_402807


namespace mutually_exclusive_not_opposite_l4028_402839

def Bag := Fin 4

def is_black : Bag → Prop :=
  fun b => b.val < 2

def Draw := Fin 2 → Bag

def exactly_one_black (draw : Draw) : Prop :=
  (is_black (draw 0) ∧ ¬is_black (draw 1)) ∨ (¬is_black (draw 0) ∧ is_black (draw 1))

def exactly_two_black (draw : Draw) : Prop :=
  is_black (draw 0) ∧ is_black (draw 1)

theorem mutually_exclusive_not_opposite :
  (∃ (draw : Draw), exactly_one_black draw) ∧
  (∃ (draw : Draw), exactly_two_black draw) ∧
  (¬∃ (draw : Draw), exactly_one_black draw ∧ exactly_two_black draw) ∧
  (∃ (draw : Draw), ¬exactly_one_black draw ∧ ¬exactly_two_black draw) :=
sorry

end mutually_exclusive_not_opposite_l4028_402839


namespace length_XX₁_l4028_402897

-- Define the triangles and circle
def triangle_DEF (DE DF : ℝ) : Prop := DE = 7 ∧ DF = 3
def inscribed_circle (F₁ : ℝ × ℝ) : Prop := sorry  -- Details of circle inscription

-- Define the second triangle XYZ
def triangle_XYZ (XY XZ : ℝ) (F₁E F₁D : ℝ) : Prop :=
  XY = F₁E ∧ XZ = F₁D

-- Define the angle bisector and point X₁
def angle_bisector (X₁ : ℝ × ℝ) : Prop := sorry  -- Details of angle bisector

-- Main theorem
theorem length_XX₁ (DE DF : ℝ) (F₁ : ℝ × ℝ) (XY XZ : ℝ) (X₁ : ℝ × ℝ) :
  triangle_DEF DE DF →
  inscribed_circle F₁ →
  triangle_XYZ XY XZ (Real.sqrt 10 - 2) (Real.sqrt 10 - 2) →
  angle_bisector X₁ →
  ∃ (XX₁ : ℝ), XX₁ = 2 * Real.sqrt 6 / 3 :=
sorry

end length_XX₁_l4028_402897


namespace equal_to_one_half_l4028_402826

theorem equal_to_one_half : 
  Real.sqrt ((1 + Real.cos (2 * Real.pi / 3)) / 2) = 1 / 2 := by
  sorry

end equal_to_one_half_l4028_402826


namespace book_pages_proof_l4028_402891

theorem book_pages_proof (x : ℝ) : 
  let day1_remaining := x - (x / 6 + 10)
  let day2_remaining := day1_remaining - (day1_remaining / 3 + 20)
  let day3_remaining := day2_remaining - (day2_remaining / 2 + 25)
  day3_remaining = 120 → x = 552 := by
sorry

end book_pages_proof_l4028_402891


namespace geometric_sequence_property_l4028_402884

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 1 * a 3 * a 11 = 8) : 
  a 2 * a 8 = 4 := by
  sorry

end geometric_sequence_property_l4028_402884


namespace inscribed_polygon_has_larger_area_l4028_402801

/-- A polygon is a set of points in the plane --/
def Polygon : Type := Set (ℝ × ℝ)

/-- A convex polygon is a polygon where all interior angles are less than or equal to 180 degrees --/
def ConvexPolygon (P : Polygon) : Prop := sorry

/-- A polygon is inscribed in a circle if all its vertices lie on the circle's circumference --/
def InscribedInCircle (P : Polygon) : Prop := sorry

/-- The area of a polygon --/
def PolygonArea (P : Polygon) : ℝ := sorry

/-- The side lengths of a polygon --/
def SideLengths (P : Polygon) : List ℝ := sorry

/-- Two polygons have the same side lengths --/
def SameSideLengths (P Q : Polygon) : Prop :=
  SideLengths P = SideLengths Q

theorem inscribed_polygon_has_larger_area 
  (N M : Polygon) 
  (h1 : ConvexPolygon N) 
  (h2 : ConvexPolygon M) 
  (h3 : InscribedInCircle N) 
  (h4 : SameSideLengths N M) :
  PolygonArea N > PolygonArea M :=
sorry

end inscribed_polygon_has_larger_area_l4028_402801


namespace ellipse_foci_distance_l4028_402825

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of a parallel axis ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ :=
  sorry

/-- Theorem stating the distance between foci for the given ellipse -/
theorem ellipse_foci_distance :
  ∀ (e : ParallelAxisEllipse),
    e.x_tangent = (6, 0) →
    e.y_tangent = (0, 3) →
    foci_distance e = 6 * Real.sqrt 3 :=
  sorry

end ellipse_foci_distance_l4028_402825


namespace function_equality_l4028_402899

theorem function_equality (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) : 
  ∀ n : ℕ, f n = n := by
sorry

end function_equality_l4028_402899


namespace inverse_f_at_135_l4028_402894

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x^3 + 5

-- State the theorem
theorem inverse_f_at_135 :
  ∃ (y : ℝ), f y = 135 ∧ y = (26 : ℝ)^(1/3) :=
sorry

end inverse_f_at_135_l4028_402894


namespace average_minutes_run_is_112_div_9_l4028_402810

/-- The average number of minutes run per day by all students in an elementary school -/
def average_minutes_run (third_grade_minutes fourth_grade_minutes fifth_grade_minutes : ℕ)
  (third_to_fourth_ratio fourth_to_fifth_ratio : ℕ) : ℚ :=
  let fifth_graders := 1
  let fourth_graders := fourth_to_fifth_ratio * fifth_graders
  let third_graders := third_to_fourth_ratio * fourth_graders
  let total_students := third_graders + fourth_graders + fifth_graders
  let total_minutes := third_grade_minutes * third_graders + 
                       fourth_grade_minutes * fourth_graders + 
                       fifth_grade_minutes * fifth_graders
  (total_minutes : ℚ) / total_students

theorem average_minutes_run_is_112_div_9 :
  average_minutes_run 10 18 16 3 2 = 112 / 9 := by
  sorry

end average_minutes_run_is_112_div_9_l4028_402810


namespace parabola_equation_from_hyperbola_focus_l4028_402824

theorem parabola_equation_from_hyperbola_focus : ∃ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 3 - y^2 / 6 = 1) →
  c^2 = a^2 + b^2 →
  (∀ x y : ℝ, y^2 = 4 * c * x ↔ y^2 = 12 * x) :=
by sorry

end parabola_equation_from_hyperbola_focus_l4028_402824


namespace complex_expression_equals_minus_half_minus_half_i_l4028_402837

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The complex number (1+i)^2 / (1-i)^3 -/
noncomputable def complex_expression : ℂ := (1 + i)^2 / (1 - i)^3

/-- Theorem stating that the complex expression equals -1/2 - 1/2i -/
theorem complex_expression_equals_minus_half_minus_half_i :
  complex_expression = -1/2 - 1/2 * i :=
by sorry

end complex_expression_equals_minus_half_minus_half_i_l4028_402837


namespace fibonacci_identity_l4028_402838

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Statement of the problem -/
theorem fibonacci_identity (k : ℤ) :
  (fib 785 + k) * (fib 787 + k) - (fib 786 + k)^2 = -1 := by
  sorry


end fibonacci_identity_l4028_402838


namespace virus_memory_growth_l4028_402890

theorem virus_memory_growth (initial_memory : ℕ) (growth_interval : ℕ) (final_memory : ℕ) :
  initial_memory = 2 →
  growth_interval = 3 →
  final_memory = 64 * 2^10 →
  (fun n => initial_memory * 2^n) (15 * growth_interval / growth_interval) = final_memory :=
by sorry

end virus_memory_growth_l4028_402890


namespace triangle_bd_length_l4028_402896

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the point D on AB
def D (t : Triangle) : ℝ × ℝ := sorry

-- State the theorem
theorem triangle_bd_length (t : Triangle) :
  -- Conditions
  (dist t.A t.C = 7) →
  (dist t.B t.C = 7) →
  (dist t.A (D t) = 8) →
  (dist t.C (D t) = 3) →
  -- Conclusion
  (dist t.B (D t) = 5) := by
  sorry

where
  dist : (ℝ × ℝ) → (ℝ × ℝ) → ℝ := sorry

end triangle_bd_length_l4028_402896


namespace largest_prime_factor_of_999_l4028_402853

theorem largest_prime_factor_of_999 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 999 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 999 → q ≤ p := by
  sorry

end largest_prime_factor_of_999_l4028_402853


namespace booklet_sheets_theorem_l4028_402819

/-- Given a stack of sheets folded into a booklet, this function calculates
    the number of sheets in the original stack based on the sum of page numbers on one sheet. -/
def calculate_original_sheets (sum_of_page_numbers : ℕ) : ℕ :=
  (sum_of_page_numbers - 2) / 4

/-- Theorem stating that if the sum of page numbers on one sheet is 74,
    then the original stack contained 9 sheets. -/
theorem booklet_sheets_theorem (sum_is_74 : calculate_original_sheets 74 = 9) :
  calculate_original_sheets 74 = 9 := by
  sorry

#eval calculate_original_sheets 74  -- Should output 9

end booklet_sheets_theorem_l4028_402819


namespace expensive_handcuffs_time_l4028_402888

/-- The time it takes to pick the lock on an expensive pair of handcuffs -/
def time_expensive : ℝ := 8

/-- The time it takes to pick the lock on a cheap pair of handcuffs -/
def time_cheap : ℝ := 6

/-- The number of friends to rescue -/
def num_friends : ℕ := 3

/-- The total time it takes to free all friends -/
def total_time : ℝ := 42

theorem expensive_handcuffs_time :
  time_expensive = (total_time - num_friends * time_cheap) / num_friends := by
  sorry

end expensive_handcuffs_time_l4028_402888


namespace log_difference_theorem_l4028_402867

noncomputable def logBase (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

def satisfies_condition (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧
  (∀ x ∈ Set.Icc 1 3, logBase a x ≤ logBase a 3) ∧
  (∀ x ∈ Set.Icc 1 3, logBase a x ≥ logBase a 1) ∧
  logBase a 3 - logBase a 1 = 2

theorem log_difference_theorem :
  {a : ℝ | satisfies_condition a} = {Real.sqrt 3, Real.sqrt 3 / 3} :=
sorry

end log_difference_theorem_l4028_402867


namespace integer_list_mean_mode_l4028_402868

theorem integer_list_mean_mode (y : ℕ) : 
  y > 0 ∧ y ≤ 150 →
  let l := [45, 76, 123, y, y, y]
  (l.sum / l.length : ℚ) = 2 * y →
  y = 27 := by
sorry

end integer_list_mean_mode_l4028_402868


namespace sum_of_extreme_prime_factors_1260_l4028_402863

theorem sum_of_extreme_prime_factors_1260 : ∃ (p q : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ 
  p ∣ 1260 ∧ q ∣ 1260 ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 1260 → p ≤ r ∧ r ≤ q) ∧
  p + q = 9 :=
by sorry

end sum_of_extreme_prime_factors_1260_l4028_402863


namespace person_age_l4028_402875

theorem person_age : ∃ (age : ℕ), 
  (4 * (age + 4) - 4 * (age - 4) = age) ∧ (age = 32) := by
  sorry

end person_age_l4028_402875


namespace rain_period_end_time_l4028_402816

/-- Represents time in 24-hour format -/
structure Time where
  hour : ℕ
  minute : ℕ

/-- Adds hours to a given time -/
def addHours (t : Time) (h : ℕ) : Time :=
  { hour := (t.hour + h) % 24, minute := t.minute }

theorem rain_period_end_time 
  (start : Time)
  (rain_duration : ℕ)
  (no_rain_duration : ℕ)
  (h_start : start = { hour := 9, minute := 0 })
  (h_rain : rain_duration = 2)
  (h_no_rain : no_rain_duration = 6) :
  addHours start (rain_duration + no_rain_duration) = { hour := 17, minute := 0 } :=
sorry

end rain_period_end_time_l4028_402816


namespace new_york_squares_count_l4028_402852

/-- The number of squares in New York City -/
def num_squares : ℕ := 15

/-- The total number of streetlights bought by the city council -/
def total_streetlights : ℕ := 200

/-- The number of streetlights required for each square -/
def streetlights_per_square : ℕ := 12

/-- The number of unused streetlights -/
def unused_streetlights : ℕ := 20

/-- Theorem stating that the number of squares in New York City is correct -/
theorem new_york_squares_count :
  num_squares * streetlights_per_square + unused_streetlights = total_streetlights :=
by sorry

end new_york_squares_count_l4028_402852


namespace binary_addition_subtraction_l4028_402845

theorem binary_addition_subtraction : 
  let a : ℕ := 0b1101
  let b : ℕ := 0b1010
  let c : ℕ := 0b1111
  let d : ℕ := 0b1001
  a + b - c + d = 0b11001 := by sorry

end binary_addition_subtraction_l4028_402845


namespace power_division_rule_l4028_402806

theorem power_division_rule (a : ℝ) : a^4 / a = a^3 := by
  sorry

end power_division_rule_l4028_402806


namespace triangle_area_relationship_uncertain_l4028_402855

/-- A triangle with side lengths and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

/-- Proposition: The relationship between areas of two triangles is uncertain -/
theorem triangle_area_relationship_uncertain 
  (ABC : Triangle) (A₁B₁C₁ : Triangle) 
  (h1 : ABC.a > A₁B₁C₁.a) 
  (h2 : ABC.b > A₁B₁C₁.b) 
  (h3 : ABC.c > A₁B₁C₁.c) :
  ¬ (∀ (ABC A₁B₁C₁ : Triangle), 
    ABC.a > A₁B₁C₁.a → ABC.b > A₁B₁C₁.b → ABC.c > A₁B₁C₁.c → 
    (ABC.area > A₁B₁C₁.area ∨ ABC.area < A₁B₁C₁.area ∨ ABC.area = A₁B₁C₁.area)) :=
sorry

end triangle_area_relationship_uncertain_l4028_402855


namespace lakota_new_cd_count_l4028_402820

/-- The price of a used CD in dollars -/
def used_cd_price : ℚ := 9.99

/-- The total price of Lakota's purchase in dollars -/
def lakota_total : ℚ := 127.92

/-- The total price of Mackenzie's purchase in dollars -/
def mackenzie_total : ℚ := 133.89

/-- The number of used CDs Lakota bought -/
def lakota_used : ℕ := 2

/-- The number of new CDs Mackenzie bought -/
def mackenzie_new : ℕ := 3

/-- The number of used CDs Mackenzie bought -/
def mackenzie_used : ℕ := 8

/-- The number of new CDs Lakota bought -/
def lakota_new : ℕ := 6

theorem lakota_new_cd_count :
  ∃ (new_cd_price : ℚ),
    new_cd_price * lakota_new + used_cd_price * lakota_used = lakota_total ∧
    new_cd_price * mackenzie_new + used_cd_price * mackenzie_used = mackenzie_total :=
by sorry

end lakota_new_cd_count_l4028_402820


namespace chip_price_is_two_l4028_402823

/-- The price of a packet of chips -/
def chip_price : ℝ := sorry

/-- The price of a packet of corn chips -/
def corn_chip_price : ℝ := 1.5

/-- The number of packets of chips John buys -/
def num_chips : ℕ := 15

/-- The number of packets of corn chips John buys -/
def num_corn_chips : ℕ := 10

/-- John's total budget -/
def total_budget : ℝ := 45

theorem chip_price_is_two :
  chip_price * num_chips + corn_chip_price * num_corn_chips = total_budget →
  chip_price = 2 := by sorry

end chip_price_is_two_l4028_402823


namespace folded_rectangle_EF_length_l4028_402809

-- Define the rectangle
structure Rectangle :=
  (AB : ℝ)
  (BC : ℝ)

-- Define the folded pentagon
structure FoldedPentagon :=
  (rect : Rectangle)
  (EF : ℝ)

-- Theorem statement
theorem folded_rectangle_EF_length 
  (rect : Rectangle) 
  (pent : FoldedPentagon) : 
  rect.AB = 4 → 
  rect.BC = 8 → 
  pent.rect = rect → 
  pent.EF = 4 := by
sorry

end folded_rectangle_EF_length_l4028_402809


namespace prime_pairs_congruence_l4028_402865

theorem prime_pairs_congruence (p : Nat) : Prime p →
  (∃! n : Nat, n = (Finset.filter (fun pair : Nat × Nat =>
    0 ≤ pair.1 ∧ pair.1 ≤ p ∧
    0 ≤ pair.2 ∧ pair.2 ≤ p ∧
    (pair.2 ^ 2) % p = ((pair.1 ^ 3) - pair.1) % p)
    (Finset.product (Finset.range (p + 1)) (Finset.range (p + 1)))).card ∧ n = p) ↔
  (p = 2 ∨ p % 4 = 3) :=
by sorry

end prime_pairs_congruence_l4028_402865


namespace trig_inequalities_l4028_402842

theorem trig_inequalities :
  (Real.cos (3 * Real.pi / 5) > Real.cos (-4 * Real.pi / 5)) ∧
  (Real.sin (Real.pi / 10) < Real.cos (Real.pi / 10)) := by
  sorry

end trig_inequalities_l4028_402842


namespace recipe_calculation_l4028_402812

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℚ
  flour : ℚ
  sugar : ℚ

/-- Calculates the required amount of an ingredient based on the ratio and the amount of sugar used -/
def calculateAmount (ratio : RecipeRatio) (sugarAmount : ℚ) (partRatio : ℚ) : ℚ :=
  (sugarAmount / ratio.sugar) * partRatio

/-- Proves that given a recipe with a butter:flour:sugar ratio of 1:6:4 and using 10 cups of sugar,
    the required amounts of butter and flour are 2.5 cups and 15 cups, respectively -/
theorem recipe_calculation (ratio : RecipeRatio) (sugarAmount : ℚ) :
  ratio.butter = 1 → ratio.flour = 6 → ratio.sugar = 4 → sugarAmount = 10 →
  calculateAmount ratio sugarAmount ratio.butter = 5/2 ∧
  calculateAmount ratio sugarAmount ratio.flour = 15 :=
by sorry

end recipe_calculation_l4028_402812


namespace min_trips_for_given_weights_l4028_402803

-- Define the list of people's weights
def weights : List ℕ := [130, 60, 61, 65, 68, 70, 79, 81, 83, 87, 90, 91, 95]

-- Define the elevator capacity
def capacity : ℕ := 175

-- Function to calculate the minimum number of trips
def min_trips (weights : List ℕ) (capacity : ℕ) : ℕ := sorry

-- Theorem stating that the minimum number of trips is 7
theorem min_trips_for_given_weights :
  min_trips weights capacity = 7 := by sorry

end min_trips_for_given_weights_l4028_402803


namespace stairs_in_building_correct_stairs_count_l4028_402813

theorem stairs_in_building (ned_speed : ℕ) (bomb_time_left : ℕ) (time_spent_running : ℕ) (diffuse_time : ℕ) : ℕ :=
  let total_run_time := time_spent_running + (bomb_time_left - diffuse_time)
  total_run_time / ned_speed

theorem correct_stairs_count : stairs_in_building 11 72 165 17 = 20 := by
  sorry

end stairs_in_building_correct_stairs_count_l4028_402813


namespace coefficient_x3y4_expansion_l4028_402830

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

-- Define the expansion term
def expansionTerm (n k : ℕ) (x y : ℚ) : ℚ :=
  binomial n k * (x ^ k) * (y ^ (n - k))

-- Theorem statement
theorem coefficient_x3y4_expansion :
  let n : ℕ := 9
  let k : ℕ := 3
  let x : ℚ := 2/3
  let y : ℚ := -3/4
  expansionTerm n k x y = 441/992 := by
sorry

end coefficient_x3y4_expansion_l4028_402830


namespace binary_predecessor_and_successor_l4028_402804

def binary_number : ℕ := 84  -- 1010100₂ in decimal

theorem binary_predecessor_and_successor :
  (binary_number - 1 = 83) ∧ (binary_number + 1 = 85) := by
  sorry

-- Helper function to convert decimal to binary string (for reference)
def to_binary (n : ℕ) : String :=
  if n = 0 then "0"
  else
    let rec aux (m : ℕ) (acc : String) : String :=
      if m = 0 then acc
      else aux (m / 2) (toString (m % 2) ++ acc)
    aux n ""

-- These computations are to verify the binary representations
#eval to_binary binary_number        -- Should output "1010100"
#eval to_binary (binary_number - 1)  -- Should output "1010011"
#eval to_binary (binary_number + 1)  -- Should output "1010101"

end binary_predecessor_and_successor_l4028_402804


namespace non_fiction_count_l4028_402889

/-- The number of fiction books -/
def fiction_books : ℕ := 5

/-- The number of ways to select 2 fiction and 2 non-fiction books -/
def selection_ways : ℕ := 150

/-- Combination function -/
def C (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of non-fiction books -/
def non_fiction_books : ℕ := 6

theorem non_fiction_count : 
  C fiction_books 2 * C non_fiction_books 2 = selection_ways := by sorry

end non_fiction_count_l4028_402889


namespace line_BC_equation_l4028_402850

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem conditions
def problem_conditions (triangle : Triangle) (altitude1 altitude2 : Line) : Prop :=
  -- First altitude: x + y = 0
  altitude1.a = 1 ∧ altitude1.b = 1 ∧ altitude1.c = 0 ∧
  -- Second altitude: 2x - 3y + 1 = 0
  altitude2.a = 2 ∧ altitude2.b = -3 ∧ altitude2.c = 1 ∧
  -- Point A is (1, 2)
  triangle.A = (1, 2)

-- Theorem statement
theorem line_BC_equation (triangle : Triangle) (altitude1 altitude2 : Line) :
  problem_conditions triangle altitude1 altitude2 →
  ∃ (line_BC : Line), line_BC.a = 2 ∧ line_BC.b = 3 ∧ line_BC.c = 7 :=
by
  sorry

end line_BC_equation_l4028_402850


namespace difference_of_squares_65_35_l4028_402821

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end difference_of_squares_65_35_l4028_402821


namespace probability_at_least_one_contract_probability_at_least_one_contract_proof_l4028_402872

theorem probability_at_least_one_contract 
  (p_hardware : ℚ) 
  (p_not_software : ℚ) 
  (p_both : ℚ) 
  (h1 : p_hardware = 3/4) 
  (h2 : p_not_software = 5/9) 
  (h3 : p_both = 71/180) -- 0.3944444444444444 ≈ 71/180
  : ℚ :=
  29/36

theorem probability_at_least_one_contract_proof 
  (p_hardware : ℚ) 
  (p_not_software : ℚ) 
  (p_both : ℚ) 
  (h1 : p_hardware = 3/4) 
  (h2 : p_not_software = 5/9) 
  (h3 : p_both = 71/180)
  : probability_at_least_one_contract p_hardware p_not_software p_both h1 h2 h3 = 29/36 := by
  sorry

end probability_at_least_one_contract_probability_at_least_one_contract_proof_l4028_402872


namespace cars_meeting_time_l4028_402859

/-- Two cars traveling from opposite ends of a highway meet after a certain time. -/
theorem cars_meeting_time
  (highway_length : ℝ)
  (car1_speed : ℝ)
  (car2_speed : ℝ)
  (h1 : highway_length = 175)
  (h2 : car1_speed = 25)
  (h3 : car2_speed = 45) :
  (highway_length / (car1_speed + car2_speed)) = 2.5 := by
  sorry


end cars_meeting_time_l4028_402859


namespace x_plus_reciprocal_x_l4028_402840

theorem x_plus_reciprocal_x (x : ℝ) 
  (h1 : x^3 + 1/x^3 = 110) 
  (h2 : (x + 1/x)^2 - 2*x - 2/x = 38) : 
  x + 1/x = 5 := by
  sorry

end x_plus_reciprocal_x_l4028_402840


namespace science_club_enrollment_l4028_402805

theorem science_club_enrollment (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : biology = 40)
  (h3 : chemistry = 35)
  (h4 : both = 25) :
  total - (biology + chemistry - both) = 10 := by
  sorry

end science_club_enrollment_l4028_402805


namespace fund_raising_ratio_l4028_402862

def fund_raising (goal : ℕ) (ken_collection : ℕ) (excess : ℕ) : Prop :=
  ∃ (mary_collection scott_collection : ℕ),
    mary_collection = 5 * ken_collection ∧
    ∃ (k : ℕ), mary_collection = k * scott_collection ∧
    mary_collection + scott_collection + ken_collection = goal + excess ∧
    mary_collection / scott_collection = 3

theorem fund_raising_ratio :
  fund_raising 4000 600 600 :=
sorry

end fund_raising_ratio_l4028_402862


namespace largest_n_perfect_cube_l4028_402833

theorem largest_n_perfect_cube (n : ℕ) : n = 497 ↔ 
  (n < 500 ∧ 
   ∃ m : ℕ, 6048 * 28^n = m^3 ∧ 
   ∀ k : ℕ, k < 500 ∧ k > n → ¬∃ l : ℕ, 6048 * 28^k = l^3) :=
by sorry

end largest_n_perfect_cube_l4028_402833


namespace abs_neg_one_sixth_gt_neg_one_seventh_l4028_402834

theorem abs_neg_one_sixth_gt_neg_one_seventh : |-(1/6)| > -(1/7) := by
  sorry

end abs_neg_one_sixth_gt_neg_one_seventh_l4028_402834


namespace ages_cube_sum_l4028_402881

theorem ages_cube_sum (r j m : ℕ) : 
  (5 * r + 2 * j = 3 * m) →
  (3 * m^2 + 2 * j^2 = 5 * r^2) →
  (Nat.gcd r j = 1 ∧ Nat.gcd j m = 1 ∧ Nat.gcd r m = 1) →
  r^3 + j^3 + m^3 = 3 :=
by sorry

end ages_cube_sum_l4028_402881


namespace min_value_constraint_min_value_achieved_l4028_402858

theorem min_value_constraint (x y : ℝ) (h : 2 * x + 8 * y = 3) :
  x^2 + 4 * y^2 - 2 * x ≥ -19/20 := by
  sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, 2 * x + 8 * y = 3 ∧ x^2 + 4 * y^2 - 2 * x < -19/20 + ε := by
  sorry

end min_value_constraint_min_value_achieved_l4028_402858


namespace brian_video_time_l4028_402822

/-- The duration of Brian's animal video watching session -/
def total_video_time (cat_video_duration : ℕ) : ℕ :=
  let dog_video_duration := 2 * cat_video_duration
  let first_two_videos_duration := cat_video_duration + dog_video_duration
  let gorilla_video_duration := 2 * first_two_videos_duration
  cat_video_duration + dog_video_duration + gorilla_video_duration

/-- Theorem stating that Brian spends 36 minutes watching animal videos -/
theorem brian_video_time : total_video_time 4 = 36 := by
  sorry

end brian_video_time_l4028_402822


namespace candidates_calculation_l4028_402835

theorem candidates_calculation (total_candidates : ℕ) : 
  (total_candidates * 6 / 100 : ℚ) + 83 = (total_candidates * 7 / 100 : ℚ) → 
  total_candidates = 8300 := by
  sorry

end candidates_calculation_l4028_402835


namespace locus_of_sine_zero_l4028_402864

theorem locus_of_sine_zero (x y : ℝ) : 
  Real.sin (x + y) = 0 ↔ ∃ k : ℤ, x + y = k * Real.pi := by sorry

end locus_of_sine_zero_l4028_402864


namespace girls_in_college_l4028_402815

theorem girls_in_college (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 312 →
  boys + girls = total →
  8 * girls = 5 * boys →
  girls = 120 := by
sorry

end girls_in_college_l4028_402815


namespace soap_brand_usage_l4028_402861

theorem soap_brand_usage (total : ℕ) (neither : ℕ) (both : ℕ) :
  total = 180 →
  neither = 80 →
  both = 10 →
  ∃ (only_A only_B : ℕ),
    total = only_A + only_B + both + neither ∧
    only_B = 3 * both ∧
    only_A = 60 := by
  sorry

end soap_brand_usage_l4028_402861


namespace parallel_vectors_subtraction_l4028_402886

/-- Given vectors a and b in ℝ², where a is parallel to b, prove that 2a - b = (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, 4]
  (∃ (k : ℝ), a = k • b) →
  (2 • a - b) = ![4, -8] := by
sorry

end parallel_vectors_subtraction_l4028_402886


namespace triangle_trig_expression_l4028_402817

theorem triangle_trig_expression (D E F : Real) (DE DF EF : Real) : 
  DE = 8 → DF = 10 → EF = 6 → 
  (Real.cos ((D - E) / 2) / Real.sin (F / 2)) - (Real.sin ((D - E) / 2) / Real.cos (F / 2)) = 5 / 2 := by
  sorry

end triangle_trig_expression_l4028_402817


namespace correct_calculation_l4028_402869

theorem correct_calculation (x : ℤ) (h : x - 48 = 52) : x + 48 = 148 := by
  sorry

end correct_calculation_l4028_402869


namespace constant_c_value_l4028_402877

theorem constant_c_value (b c : ℝ) : 
  (∀ x : ℝ, (x + 2) * (x + b) = x^2 + c*x + 6) → c = 5 := by
  sorry

end constant_c_value_l4028_402877


namespace number_relationship_theorem_l4028_402876

theorem number_relationship_theorem :
  ∃ (x y a : ℝ), x = 6 * y - a ∧ x + y = 38 ∧
  (∀ (x' y' : ℝ), x' = 6 * y' - a ∧ x' + y' = 38 → x' = x ∧ y' = y → a = a) := by
  sorry

end number_relationship_theorem_l4028_402876


namespace werewolf_unreachable_l4028_402885

def is_black (x y : Int) : Bool :=
  x % 2 = y % 2

def possible_moves : List (Int × Int) :=
  [(1, 2), (2, -1), (-1, -2), (-2, 1)]

def reachable (start_x start_y end_x end_y : Int) : Prop :=
  ∃ (n : Nat), ∃ (moves : List (Int × Int)),
    moves.length = n ∧
    moves.all (λ m => m ∈ possible_moves) ∧
    (moves.foldl (λ (x, y) (dx, dy) => (x + dx, y + dy)) (start_x, start_y) = (end_x, end_y))

theorem werewolf_unreachable :
  ¬(reachable 26 10 42 2017) :=
by sorry

end werewolf_unreachable_l4028_402885


namespace tshirt_shop_profit_l4028_402860

theorem tshirt_shop_profit : 
  let profit_per_shirt : ℚ := 9
  let cost_per_shirt : ℚ := 4
  let num_shirts : ℕ := 245
  let discount_rate : ℚ := 1/5

  let original_price : ℚ := profit_per_shirt + cost_per_shirt
  let discounted_price : ℚ := original_price * (1 - discount_rate)
  let total_revenue : ℚ := (discounted_price * num_shirts : ℚ)
  let total_cost : ℚ := (cost_per_shirt * num_shirts : ℚ)
  let total_profit : ℚ := total_revenue - total_cost

  total_profit = 1568 := by sorry

end tshirt_shop_profit_l4028_402860


namespace quadratic_bounded_values_l4028_402898

/-- A quadratic function f(x) = ax^2 + bx + c where a > 100 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_bounded_values (a b c : ℝ) (ha : a > 100) :
  ∃ (n : ℕ), n ≤ 2 ∧
  ∀ (S : Finset ℤ), (∀ x ∈ S, |QuadraticFunction a b c x| ≤ 50) →
  Finset.card S ≤ n :=
sorry

end quadratic_bounded_values_l4028_402898


namespace umar_age_l4028_402866

/-- Given the ages of Ali, Yusaf, and Umar, prove Umar's age -/
theorem umar_age (ali_age yusaf_age umar_age : ℕ) : 
  ali_age = 8 →
  ali_age = yusaf_age + 3 →
  umar_age = 2 * yusaf_age →
  umar_age = 10 := by
sorry

end umar_age_l4028_402866


namespace nell_gave_jeff_cards_nell_gave_jeff_cards_proof_l4028_402854

/-- Given that Nell initially had 304 baseball cards and now has 276 cards left,
    prove that she gave 28 cards to Jeff. -/
theorem nell_gave_jeff_cards : ℕ → ℕ → ℕ → Prop :=
  fun initial_cards remaining_cards cards_given =>
    initial_cards = 304 →
    remaining_cards = 276 →
    cards_given = initial_cards - remaining_cards →
    cards_given = 28

/-- Proof of the theorem -/
theorem nell_gave_jeff_cards_proof : nell_gave_jeff_cards 304 276 28 := by
  sorry

end nell_gave_jeff_cards_nell_gave_jeff_cards_proof_l4028_402854


namespace sqrt_division_equality_l4028_402882

theorem sqrt_division_equality : Real.sqrt 10 / Real.sqrt 5 = Real.sqrt 2 := by
  sorry

end sqrt_division_equality_l4028_402882


namespace divisors_sum_product_l4028_402874

theorem divisors_sum_product (n a b : ℕ) : 
  n ≥ 1 → 
  a > 0 → 
  b > 0 → 
  n % a = 0 → 
  n % b = 0 → 
  a + b + a * b = n → 
  a = b := by
sorry

end divisors_sum_product_l4028_402874


namespace not_divisible_by_three_l4028_402851

theorem not_divisible_by_three (n : ℤ) : ¬(3 ∣ (n^2 + 1)) := by
  sorry

end not_divisible_by_three_l4028_402851
