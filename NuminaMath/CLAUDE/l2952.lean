import Mathlib

namespace yahs_to_bahs_conversion_l2952_295285

/-- Given the conversion rates between bahs, rahs, and yahs, 
    prove that 1500 yahs are equivalent to 500 bahs. -/
theorem yahs_to_bahs_conversion 
  (bah_to_rah : (20 : ℚ) / 36 = 1 / (36 / 20)) 
  (rah_to_yah : (12 : ℚ) / 20 = 1 / (20 / 12)) : 
  (1500 : ℚ) * (12 / 20) * (20 / 36) = 500 :=
sorry

end yahs_to_bahs_conversion_l2952_295285


namespace chairs_per_row_l2952_295247

theorem chairs_per_row (total_rows : ℕ) (occupied_seats : ℕ) (unoccupied_seats : ℕ) :
  total_rows = 40 →
  occupied_seats = 790 →
  unoccupied_seats = 10 →
  (occupied_seats + unoccupied_seats) / total_rows = 20 :=
by
  sorry

end chairs_per_row_l2952_295247


namespace tan_sqrt_two_identity_l2952_295229

theorem tan_sqrt_two_identity (α : Real) (h : Real.tan α = Real.sqrt 2) :
  1 + Real.sin (2 * α) + (Real.cos α)^2 = (4 + Real.sqrt 2) / 3 := by
  sorry

end tan_sqrt_two_identity_l2952_295229


namespace net_income_for_130_tax_l2952_295240

/-- Calculates the net income after tax given a pre-tax income -/
def net_income (pre_tax_income : ℝ) : ℝ :=
  pre_tax_income - ((pre_tax_income - 800) * 0.2)

/-- Theorem stating that for a pre-tax income resulting in 130 yuan tax, the net income is 1320 yuan -/
theorem net_income_for_130_tax :
  ∃ (pre_tax_income : ℝ),
    (pre_tax_income - 800) * 0.2 = 130 ∧
    net_income pre_tax_income = 1320 :=
by sorry

end net_income_for_130_tax_l2952_295240


namespace basketball_card_price_basketball_card_price_proof_l2952_295216

/-- The price of a basketball card pack given the following conditions:
  * Olivia bought 2 packs of basketball cards
  * She bought 5 decks of baseball cards at $4 each
  * She had one $50 bill and received $24 in change
-/
theorem basketball_card_price : ℝ :=
  let baseball_card_price : ℝ := 4
  let baseball_card_count : ℕ := 5
  let basketball_card_count : ℕ := 2
  let total_money : ℝ := 50
  let change : ℝ := 24
  let spent_money : ℝ := total_money - change
  let baseball_total : ℝ := baseball_card_price * baseball_card_count
  3

theorem basketball_card_price_proof :
  let baseball_card_price : ℝ := 4
  let baseball_card_count : ℕ := 5
  let basketball_card_count : ℕ := 2
  let total_money : ℝ := 50
  let change : ℝ := 24
  let spent_money : ℝ := total_money - change
  let baseball_total : ℝ := baseball_card_price * baseball_card_count
  basketball_card_price = 3 := by
  sorry

end basketball_card_price_basketball_card_price_proof_l2952_295216


namespace common_chord_equation_l2952_295232

/-- The equation of the line where the common chord of two circles lies -/
theorem common_chord_equation (r : ℝ) (h : r > 0) :
  ∃ (ρ θ : ℝ), (ρ = r ∨ ρ = -2 * r * Real.sin (θ + π/4)) →
  Real.sqrt 2 * ρ * (Real.sin θ + Real.cos θ) = -r :=
sorry

end common_chord_equation_l2952_295232


namespace three_squares_decomposition_l2952_295208

theorem three_squares_decomposition (n : ℤ) (h : n > 5) :
  3 * (n - 1)^2 + 32 = (n - 5)^2 + (n - 1)^2 + (n + 3)^2 := by
  sorry

end three_squares_decomposition_l2952_295208


namespace exam_students_count_l2952_295222

theorem exam_students_count (N : ℕ) (T : ℕ) : 
  T = 88 * N ∧ 
  T - 8 * 50 = 92 * (N - 8) ∧ 
  T - 8 * 50 - 100 = 92 * (N - 9) → 
  N = 84 := by
sorry

end exam_students_count_l2952_295222


namespace condo_units_per_floor_l2952_295227

/-- The number of units on each regular floor in a condo development -/
def units_per_regular_floor (total_floors : ℕ) (penthouse_floors : ℕ) (units_per_penthouse : ℕ) (total_units : ℕ) : ℕ :=
  (total_units - penthouse_floors * units_per_penthouse) / (total_floors - penthouse_floors)

/-- Theorem stating that the number of units on each regular floor is 12 -/
theorem condo_units_per_floor :
  units_per_regular_floor 23 2 2 256 = 12 := by
  sorry

end condo_units_per_floor_l2952_295227


namespace day_relationship_l2952_295246

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific day in a year -/
structure DayInYear where
  year : ℤ
  dayNumber : ℕ

/-- Function to determine the day of the week for a given day in a year -/
def dayOfWeek (d : DayInYear) : DayOfWeek := sorry

/-- Theorem stating the relationship between specific days and their days of the week -/
theorem day_relationship (M : ℤ) :
  (dayOfWeek ⟨M, 250⟩ = DayOfWeek.Friday) →
  (dayOfWeek ⟨M + 1, 150⟩ = DayOfWeek.Friday) →
  (dayOfWeek ⟨M - 1, 50⟩ = DayOfWeek.Wednesday) := by
  sorry

end day_relationship_l2952_295246


namespace other_number_proof_l2952_295237

theorem other_number_proof (A B : ℕ+) (hcf lcm : ℕ+) : 
  hcf = 12 →
  lcm = 396 →
  A = 48 →
  Nat.gcd A.val B.val = hcf.val →
  Nat.lcm A.val B.val = lcm.val →
  B = 99 := by
sorry

end other_number_proof_l2952_295237


namespace angela_problems_count_l2952_295283

def total_problems : ℕ := 20
def martha_problems : ℕ := 2
def jenna_problems : ℕ := 4 * martha_problems - 2
def mark_problems : ℕ := jenna_problems / 2

theorem angela_problems_count : 
  total_problems - (martha_problems + jenna_problems + mark_problems) = 9 := by
  sorry

end angela_problems_count_l2952_295283


namespace heart_then_ten_probability_l2952_295267

/-- The number of cards in a double standard deck -/
def deck_size : ℕ := 104

/-- The number of hearts in a double standard deck -/
def num_hearts : ℕ := 26

/-- The number of 10s in a double standard deck -/
def num_tens : ℕ := 8

/-- The number of 10 of hearts in a double standard deck -/
def num_ten_hearts : ℕ := 2

/-- The probability of drawing a heart as the first card and a 10 as the second card -/
def prob_heart_then_ten : ℚ := 47 / 2678

theorem heart_then_ten_probability :
  prob_heart_then_ten = 
    (num_hearts - num_ten_hearts) / deck_size * num_tens / (deck_size - 1) +
    num_ten_hearts / deck_size * (num_tens - num_ten_hearts) / (deck_size - 1) :=
by sorry

end heart_then_ten_probability_l2952_295267


namespace circle_and_tangent_line_l2952_295253

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2 = 1

-- Define the circle ⊙G
def circle_G (x y r : ℝ) : Prop := (x-2)^2 + y^2 = r^2

-- Define the incircle property
def is_incircle (r : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ),
    ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ellipse C.1 C.2 ∧
    circle_G 2 0 r ∧
    A.1 = -4 -- Left vertex of ellipse

-- Define the tangent line EF
def line_EF (m b : ℝ) (x y : ℝ) : Prop := y = m*x + b

-- Define the tangency condition
def is_tangent (m b r : ℝ) : Prop :=
  abs (m*2 - b) / Real.sqrt (1 + m^2) = r

-- State the theorem
theorem circle_and_tangent_line :
  ∀ r : ℝ,
  is_incircle r →
  r = 2/3 ∧
  ∃ m b : ℝ,
    line_EF m b 0 1 ∧  -- Line passes through M(0,1)
    is_tangent m b r   -- Line is tangent to ⊙G
:= by sorry

end circle_and_tangent_line_l2952_295253


namespace largest_n_less_than_2023_l2952_295204

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 2 * n

-- Define the sequence b_n
def b (n : ℕ) : ℕ := 2^n

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℕ := n^2 + n

-- Define T_n
def T (n : ℕ) : ℕ := (n - 1) * 2^(n + 2) + 4

theorem largest_n_less_than_2023 :
  (∀ n : ℕ, S n = n^2 + n) →
  (∀ n : ℕ, b n = 2^n) →
  (∀ n : ℕ, T n = (n - 1) * 2^(n + 2) + 4) →
  (∃ m : ℕ, m = 6 ∧ T m < 2023 ∧ ∀ k > m, T k ≥ 2023) :=
by sorry

end largest_n_less_than_2023_l2952_295204


namespace infinitely_many_double_numbers_plus_one_square_not_power_of_ten_l2952_295292

theorem infinitely_many_double_numbers_plus_one_square_not_power_of_ten :
  ∀ m : ℕ, ∃ k > m, ∃ N : ℕ,
    Odd k ∧
    ∃ t : ℕ, N * (10^k + 1) + 1 = t^2 ∧
    ¬∃ n : ℕ, N * (10^k + 1) + 1 = 10^n := by
  sorry

end infinitely_many_double_numbers_plus_one_square_not_power_of_ten_l2952_295292


namespace parabola_focus_coordinates_l2952_295242

/-- 
Given a parabola defined by the equation x = a * y^2 where a ≠ 0,
prove that the coordinates of its focus are (1/(4*a), 0).
-/
theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  let parabola := {p : ℝ × ℝ | p.1 = a * p.2^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (1 / (4 * a), 0) :=
sorry

end parabola_focus_coordinates_l2952_295242


namespace female_students_count_l2952_295210

theorem female_students_count (total_students sample_size male_in_sample : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : male_in_sample = 110) : 
  total_students - (total_students * male_in_sample / sample_size) = 720 := by
  sorry

end female_students_count_l2952_295210


namespace range_of_5m_minus_n_l2952_295224

-- Define a decreasing and odd function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_decreasing : ∀ x y, x < y → f x > f y)
variable (h_odd : ∀ x, f (-x) = -f x)

-- Define the conditions on m and n
variable (m n : ℝ)
variable (h_cond1 : f m + f (n - 2) ≤ 0)
variable (h_cond2 : f (m - n - 1) ≤ 0)

-- Theorem statement
theorem range_of_5m_minus_n : 5 * m - n ≥ 7 := by
  sorry

end range_of_5m_minus_n_l2952_295224


namespace tan_plus_reciprocal_l2952_295271

theorem tan_plus_reciprocal (α : Real) : 
  Real.tan α + (Real.tan α)⁻¹ = (Real.sin α * Real.cos α)⁻¹ :=
by sorry

end tan_plus_reciprocal_l2952_295271


namespace baseball_card_value_decrease_l2952_295234

theorem baseball_card_value_decrease (x : ℝ) :
  (1 - x / 100) * (1 - 10 / 100) = 1 - 28 / 100 →
  x = 20 := by sorry

end baseball_card_value_decrease_l2952_295234


namespace factorial_fraction_equality_l2952_295289

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 48 / 7 := by
  sorry

end factorial_fraction_equality_l2952_295289


namespace triangle_median_and_altitude_l2952_295263

/-- Triangle ABC with given vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of median -/
def isMedian (t : Triangle) (l : LineEquation) : Prop :=
  -- The line passes through vertex B and the midpoint of AC
  sorry

/-- Definition of altitude -/
def isAltitude (t : Triangle) (l : LineEquation) : Prop :=
  -- The line passes through vertex A and is perpendicular to BC
  sorry

/-- Main theorem -/
theorem triangle_median_and_altitude (t : Triangle) 
    (h1 : t.A = (-5, 0))
    (h2 : t.B = (4, -4))
    (h3 : t.C = (0, 2)) :
    ∃ (median altitude : LineEquation),
      isMedian t median ∧
      isAltitude t altitude ∧
      median = LineEquation.mk 1 7 5 ∧
      altitude = LineEquation.mk 2 (-3) 10 := by
  sorry


end triangle_median_and_altitude_l2952_295263


namespace graphs_with_inverses_l2952_295200

-- Define the types of graphs
inductive GraphType
| Linear
| Parabola
| DisconnectedLinear
| Semicircle
| Cubic

-- Define a function to check if a graph has an inverse
def has_inverse (g : GraphType) : Prop :=
  match g with
  | GraphType.Linear => true
  | GraphType.Parabola => false
  | GraphType.DisconnectedLinear => true
  | GraphType.Semicircle => false
  | GraphType.Cubic => false

-- Define the specific graphs given in the problem
def graph_A : GraphType := GraphType.Linear
def graph_B : GraphType := GraphType.Parabola
def graph_C : GraphType := GraphType.DisconnectedLinear
def graph_D : GraphType := GraphType.Semicircle
def graph_E : GraphType := GraphType.Cubic

-- Theorem stating which graphs have inverses
theorem graphs_with_inverses :
  (has_inverse graph_A ∧ has_inverse graph_C) ∧
  (¬has_inverse graph_B ∧ ¬has_inverse graph_D ∧ ¬has_inverse graph_E) :=
by sorry

end graphs_with_inverses_l2952_295200


namespace absolute_value_inequality_l2952_295220

theorem absolute_value_inequality (x : ℝ) (h : x ≠ 1) :
  |((2 * x - 1) / (x - 1))| > 3 ↔ (4/5 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) := by
  sorry

end absolute_value_inequality_l2952_295220


namespace max_xy_min_reciprocal_sum_min_squared_sum_max_sqrt_sum_l2952_295219

variable (x y : ℝ)

-- Define the condition
def condition (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 2 * x + y = 1

-- Theorems to prove
theorem max_xy (h : condition x y) : 
  ∃ (m : ℝ), m = 1/8 ∧ ∀ (a b : ℝ), condition a b → a * b ≤ m :=
sorry

theorem min_reciprocal_sum (h : condition x y) :
  ∃ (m : ℝ), m = 9 ∧ ∀ (a b : ℝ), condition a b → m ≤ 2/a + 1/b :=
sorry

theorem min_squared_sum (h : condition x y) :
  ∃ (m : ℝ), m = 1/2 ∧ ∀ (a b : ℝ), condition a b → m ≤ 4*a^2 + b^2 :=
sorry

theorem max_sqrt_sum (h : condition x y) :
  ∃ (m : ℝ), m < 2 ∧ ∀ (a b : ℝ), condition a b → Real.sqrt (2*a) + Real.sqrt b ≤ m :=
sorry

end max_xy_min_reciprocal_sum_min_squared_sum_max_sqrt_sum_l2952_295219


namespace gcd_of_256_196_560_l2952_295209

theorem gcd_of_256_196_560 : Nat.gcd 256 (Nat.gcd 196 560) = 28 := by sorry

end gcd_of_256_196_560_l2952_295209


namespace intersection_A_B_complement_union_l2952_295238

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | 4 - x^2 ≤ 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

-- Theorem for (¬_U A) ∪ (¬_U B)
theorem complement_union :
  (Set.univ \ A) ∪ (Set.univ \ B) = {x : ℝ | x < -2 ∨ x > -1} := by sorry

end intersection_A_B_complement_union_l2952_295238


namespace sum_a_d_equals_negative_one_l2952_295286

theorem sum_a_d_equals_negative_one
  (a b c d : ℤ)
  (eq1 : a + b = 11)
  (eq2 : b + c = 9)
  (eq3 : c + d = 3) :
  a + d = -1 := by sorry

end sum_a_d_equals_negative_one_l2952_295286


namespace angle_conversion_l2952_295293

theorem angle_conversion (π : ℝ) :
  (12 : ℝ) * (π / 180) = π / 15 :=
by sorry

end angle_conversion_l2952_295293


namespace water_bottle_theorem_l2952_295215

def water_bottle_problem (water_A : ℝ) (extra_B : ℝ) (extra_C : ℝ) : Prop :=
  let water_B : ℝ := water_A + extra_B
  let water_C_ml : ℝ := (water_B / 10) * 1000 + extra_C
  let water_C_L : ℝ := water_C_ml / 1000
  water_C_L = 4.94

theorem water_bottle_theorem :
  water_bottle_problem 3.8 8.4 3720 := by
  sorry

end water_bottle_theorem_l2952_295215


namespace linear_combination_harmonic_l2952_295287

/-- A function is harmonic if its value at each point is the average of its values at the four neighboring points. -/
def IsHarmonic (f : ℤ → ℤ → ℝ) : Prop :=
  ∀ x y : ℤ, f x y = (f (x + 1) y + f (x - 1) y + f x (y + 1) + f x (y - 1)) / 4

/-- The theorem states that a linear combination of two harmonic functions is also harmonic. -/
theorem linear_combination_harmonic
    (f g : ℤ → ℤ → ℝ) (hf : IsHarmonic f) (hg : IsHarmonic g) (a b : ℝ) :
    IsHarmonic (fun x y ↦ a * f x y + b * g x y) := by
  sorry

end linear_combination_harmonic_l2952_295287


namespace sum_of_integers_l2952_295254

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 90) 
  (h2 : x * y = 27) : 
  x + y = 12 := by sorry

end sum_of_integers_l2952_295254


namespace lcm_problem_l2952_295221

theorem lcm_problem (a b : ℕ+) : 
  (Nat.gcd a b = 1) → 
  (∃ (a_max b_max a_min b_min : ℕ+), 
    (a_max - b_max) - (a_min - b_min) = 38 ∧ 
    ∀ (x y : ℕ+), (x - y) ≤ (a_max - b_max) ∧ (x - y) ≥ (a_min - b_min)) → 
  Nat.lcm a b = 40 := by
sorry

end lcm_problem_l2952_295221


namespace four_distinct_cuts_l2952_295206

/-- Represents a square grid with holes -/
structure GridWithHoles :=
  (size : ℕ)
  (holes : List (ℕ × ℕ))

/-- Represents a cut on the grid -/
inductive Cut
  | Vertical : ℕ → Cut
  | Horizontal : ℕ → Cut
  | Diagonal : Bool → Cut

/-- Checks if two parts resulting from a cut are congruent -/
def areCongruentParts (g : GridWithHoles) (c : Cut) : Bool :=
  sorry

/-- Checks if two cuts result in different congruent parts -/
def areDifferentCuts (g : GridWithHoles) (c1 c2 : Cut) : Bool :=
  sorry

/-- Theorem: There are at least four distinct ways to cut a 4x4 grid with two symmetrical holes into congruent parts -/
theorem four_distinct_cuts (g : GridWithHoles) 
  (h1 : g.size = 4)
  (h2 : g.holes = [(1, 1), (2, 2)]) : 
  ∃ (c1 c2 c3 c4 : Cut),
    areCongruentParts g c1 ∧
    areCongruentParts g c2 ∧
    areCongruentParts g c3 ∧
    areCongruentParts g c4 ∧
    areDifferentCuts g c1 c2 ∧
    areDifferentCuts g c1 c3 ∧
    areDifferentCuts g c1 c4 ∧
    areDifferentCuts g c2 c3 ∧
    areDifferentCuts g c2 c4 ∧
    areDifferentCuts g c3 c4 :=
  sorry

end four_distinct_cuts_l2952_295206


namespace ellipse_foci_l2952_295251

/-- An ellipse defined by parametric equations -/
structure ParametricEllipse where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The foci of an ellipse -/
structure EllipseFoci where
  x : ℝ
  y : ℝ

/-- Theorem: The foci of the ellipse defined by x = 3cos(θ) and y = 5sin(θ) are (0, ±4) -/
theorem ellipse_foci (e : ParametricEllipse) 
    (hx : e.x = fun θ => 3 * Real.cos θ)
    (hy : e.y = fun θ => 5 * Real.sin θ) :
  ∃ (f₁ f₂ : EllipseFoci), f₁.x = 0 ∧ f₁.y = 4 ∧ f₂.x = 0 ∧ f₂.y = -4 :=
sorry

end ellipse_foci_l2952_295251


namespace expected_value_twelve_sided_die_l2952_295223

/-- A twelve-sided die with faces numbered from 1 to 12 -/
structure TwelveSidedDie :=
  (faces : Finset ℕ)
  (face_count : faces.card = 12)
  (face_range : ∀ n, n ∈ faces ↔ 1 ≤ n ∧ n ≤ 12)

/-- The expected value of a roll of a twelve-sided die -/
def expected_value (d : TwelveSidedDie) : ℚ :=
  (d.faces.sum id) / 12

/-- Theorem: The expected value of a roll of a twelve-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_twelve_sided_die :
  ∀ d : TwelveSidedDie, expected_value d = 13/2 :=
sorry

end expected_value_twelve_sided_die_l2952_295223


namespace square_sum_given_product_and_sum_l2952_295276

theorem square_sum_given_product_and_sum (r s : ℝ) 
  (h1 : r * s = 16) 
  (h2 : r + s = 8) : 
  r^2 + s^2 = 32 := by
  sorry

end square_sum_given_product_and_sum_l2952_295276


namespace batsman_average_after_15th_inning_l2952_295249

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (score : ℕ) : ℚ :=
  (stats.totalScore + score : ℚ) / (stats.innings + 1)

/-- Theorem: A batsman's new average after the 15th inning is 33 -/
theorem batsman_average_after_15th_inning
  (stats : BatsmanStats)
  (h1 : stats.innings = 14)
  (h2 : newAverage stats 75 = stats.average + 3)
  : newAverage stats 75 = 33 := by
  sorry


end batsman_average_after_15th_inning_l2952_295249


namespace initial_gum_count_l2952_295259

/-- The number of gum pieces Adrianna had initially -/
def initial_gum : ℕ := sorry

/-- The number of gum pieces Adrianna bought -/
def bought_gum : ℕ := 3

/-- The number of friends Adrianna gave gum to -/
def friends : ℕ := 11

/-- The number of gum pieces Adrianna has left -/
def remaining_gum : ℕ := 2

/-- Theorem stating that the initial number of gum pieces was 10 -/
theorem initial_gum_count : initial_gum = 10 := by sorry

end initial_gum_count_l2952_295259


namespace largest_divisible_n_l2952_295255

theorem largest_divisible_n : ∃ (n : ℕ), n = 15544 ∧ 
  (∀ m : ℕ, m > n → ¬(n + 26 ∣ n^3 + 2006)) ∧
  (n + 26 ∣ n^3 + 2006) :=
by sorry

end largest_divisible_n_l2952_295255


namespace village_population_l2952_295260

theorem village_population (initial_population : ℕ) 
  (h1 : initial_population = 4599) :
  let died := (initial_population : ℚ) * (1/10)
  let remained_after_death := initial_population - ⌊died⌋
  let left := (remained_after_death : ℚ) * (1/5)
  initial_population - ⌊died⌋ - ⌊left⌋ = 3312 := by
  sorry

end village_population_l2952_295260


namespace trajectory_equation_l2952_295244

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- The point P -/
def P : ℝ × ℝ := (2, 2)

/-- A point is on the trajectory if it's the midpoint of a line segment AB,
    where A and B are intersection points of a line through P and the ellipse -/
def on_trajectory (M : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    (∃ (t : ℝ), A = P + t • (B - P)) ∧
    M = (A + B) / 2

/-- The theorem stating the equation of the trajectory -/
theorem trajectory_equation (x y : ℝ) :
  on_trajectory (x, y) → (x - 1)^2 + 2*(y - 1)^2 = 3 :=
sorry

end trajectory_equation_l2952_295244


namespace gcd_840_1764_gcd_98_63_l2952_295214

-- Part 1: GCD of 840 and 1764
theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by sorry

-- Part 2: GCD of 98 and 63
theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by sorry

end gcd_840_1764_gcd_98_63_l2952_295214


namespace range_of_a_l2952_295235

def p (x : ℝ) : Prop := |4*x - 3| ≤ 1

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

theorem range_of_a :
  (∀ x a : ℝ, ¬(p x) → ¬(q x a)) ∧
  (∃ x a : ℝ, ¬(q x a) ∧ p x) →
  ∀ a : ℝ, (0 ≤ a ∧ a ≤ 1/2) ↔ (∀ x : ℝ, p x → q x a) ∧ (∃ x : ℝ, q x a ∧ ¬(p x)) :=
by sorry

end range_of_a_l2952_295235


namespace min_value_and_inequality_l2952_295284

def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

theorem min_value_and_inequality :
  (∃ (M : ℝ), (∀ (m : ℝ), (∃ (x₀ : ℝ), f x₀ ≤ m) → M ≤ m) ∧ (∃ (x₀ : ℝ), f x₀ ≤ M) ∧ M = 4) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → 3*a + b = 4 → 3/b + 1/a ≥ 3) :=
by sorry

end min_value_and_inequality_l2952_295284


namespace min_value_fraction_l2952_295256

theorem min_value_fraction (x : ℝ) (h : x > 8) : 
  x^2 / (x - 8)^2 ≥ 1 ∧ ∀ ε > 0, ∃ y > 8, y^2 / (y - 8)^2 < 1 + ε :=
sorry

end min_value_fraction_l2952_295256


namespace train_length_calculation_l2952_295277

/-- The length of a train that crosses an electric pole in a given time at a given speed. -/
def trainLength (crossingTime : ℝ) (speed : ℝ) : ℝ :=
  crossingTime * speed

/-- Theorem stating that a train crossing an electric pole in 10 seconds at 108 m/s has a length of 1080 meters. -/
theorem train_length_calculation :
  trainLength 10 108 = 1080 := by
  sorry

end train_length_calculation_l2952_295277


namespace sin_translation_left_l2952_295273

/-- Translating the graph of y = sin(2x) to the left by π/3 units results in y = sin(2x + 2π/3) -/
theorem sin_translation_left (x : ℝ) : 
  let f (t : ℝ) := Real.sin (2 * t)
  let g (t : ℝ) := f (t + π/3)
  g x = Real.sin (2 * x + 2 * π/3) := by
  sorry

end sin_translation_left_l2952_295273


namespace negation_existence_equivalence_l2952_295207

theorem negation_existence_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔ (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) := by
  sorry

end negation_existence_equivalence_l2952_295207


namespace quadratic_sum_of_squares_l2952_295280

/-- A quadratic function f(x) = x^2 + ax + b satisfying certain conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  root_condition : ∃ (p q : ℤ), p * q = 2 ∧ p + q = -a
  functional_equation : ∀ (x : ℝ), x ≠ 0 → 
    (x + 1/x)^2 + a * (x + 1/x) + b = (x^2 + a*x + b) + ((1/x)^2 + a*(1/x) + b)

/-- The main theorem stating that a^2 + b^2 = 13 for the given quadratic function -/
theorem quadratic_sum_of_squares (f : QuadraticFunction) : f.a^2 + f.b^2 = 13 := by
  sorry

end quadratic_sum_of_squares_l2952_295280


namespace min_value_expression_l2952_295233

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a * b * c) ≥ 216 := by
  sorry

end min_value_expression_l2952_295233


namespace equation_solution_l2952_295264

theorem equation_solution : ∃ m : ℚ, (24 / (3 / 2) = (24 / 3) / m) ∧ m = 1/2 := by
  sorry

end equation_solution_l2952_295264


namespace right_triangle_incircle_area_ratio_l2952_295282

theorem right_triangle_incircle_area_ratio 
  (h r : ℝ) 
  (h_pos : h > 0) 
  (r_pos : r > 0) : 
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 0 ∧ 
    x^2 + y^2 = h^2 ∧ 
    (π * r^2) / ((1/2) * x * y) = π * r / (h + r) :=
sorry

end right_triangle_incircle_area_ratio_l2952_295282


namespace trapezoid_area_l2952_295248

/-- Represents a rectangle PQRS with a trapezoid TURS inside it -/
structure RectangleWithTrapezoid where
  /-- Length of the rectangle PQRS -/
  length : ℝ
  /-- Width of the rectangle PQRS -/
  width : ℝ
  /-- Distance from P to T (same as distance from Q to U) -/
  side_length : ℝ
  /-- Area of rectangle PQRS is 24 -/
  area_eq : length * width = 24
  /-- T and U are on the top side of PQRS -/
  side_constraint : side_length < length

/-- The area of trapezoid TURS is 16 square units -/
theorem trapezoid_area (rect : RectangleWithTrapezoid) : 
  rect.width * (rect.length - 2 * rect.side_length) + 2 * (rect.side_length * rect.width / 2) = 16 := by
  sorry

#check trapezoid_area

end trapezoid_area_l2952_295248


namespace factorization_equality_l2952_295281

theorem factorization_equality (m n : ℝ) : m^2 * n - n = n * (m + 1) * (m - 1) := by
  sorry

end factorization_equality_l2952_295281


namespace max_value_theorem_l2952_295217

theorem max_value_theorem (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0) 
  (h2 : b - a - 1 ≤ 0) 
  (h3 : a ≤ 1) : 
  (∀ x y : ℝ, x + y - 2 ≥ 0 → y - x - 1 ≤ 0 → x ≤ 1 → 
    (x + 2*y) / (2*x + y) ≤ (a + 2*b) / (2*a + b)) ∧ 
  (a + 2*b) / (2*a + b) = 7/5 := by
sorry

end max_value_theorem_l2952_295217


namespace min_reciprocal_sum_l2952_295275

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) :
  (1 / x + 1 / y) ≥ 1 / 3 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 12 ∧ 1 / x + 1 / y = 1 / 3 :=
sorry

end min_reciprocal_sum_l2952_295275


namespace handshakes_in_exhibition_l2952_295288

/-- Represents a mixed-doubles tennis exhibition -/
structure MixedDoublesExhibition where
  num_teams : Nat
  players_per_team : Nat

/-- Calculates the total number of handshakes in a mixed-doubles tennis exhibition -/
def total_handshakes (exhibition : MixedDoublesExhibition) : Nat :=
  let total_players := exhibition.num_teams * exhibition.players_per_team
  let handshakes_per_player := total_players - 2  -- Exclude self and partner
  (total_players * handshakes_per_player) / 2

/-- Theorem stating that the total number of handshakes in the given exhibition is 24 -/
theorem handshakes_in_exhibition :
  ∃ (exhibition : MixedDoublesExhibition),
    exhibition.num_teams = 4 ∧
    exhibition.players_per_team = 2 ∧
    total_handshakes exhibition = 24 := by
  sorry

end handshakes_in_exhibition_l2952_295288


namespace abc_inequality_l2952_295231

theorem abc_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h_sum : a^2 + b^2 + c^2 + a*b*c = 4) : 
  0 ≤ a*b + b*c + c*a - a*b*c ∧ a*b + b*c + c*a - a*b*c ≤ 2 := by
  sorry

end abc_inequality_l2952_295231


namespace fifteen_switches_connections_l2952_295203

/-- The number of unique connections in a network of switches -/
def uniqueConnections (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a network of 15 switches, where each switch connects to 
    exactly 4 other switches, the total number of unique connections is 30. -/
theorem fifteen_switches_connections : 
  uniqueConnections 15 4 = 30 := by
  sorry

end fifteen_switches_connections_l2952_295203


namespace trigonometric_identity_l2952_295269

theorem trigonometric_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin y ^ 2 := by
  sorry

end trigonometric_identity_l2952_295269


namespace third_plane_passenger_count_l2952_295213

/-- The number of passengers on the third plane -/
def third_plane_passengers : ℕ := 40

/-- The speed of an empty plane in MPH -/
def empty_plane_speed : ℕ := 600

/-- The speed reduction per passenger in MPH -/
def speed_reduction_per_passenger : ℕ := 2

/-- The number of passengers on the first plane -/
def first_plane_passengers : ℕ := 50

/-- The number of passengers on the second plane -/
def second_plane_passengers : ℕ := 60

/-- The average speed of the three planes in MPH -/
def average_speed : ℕ := 500

theorem third_plane_passenger_count :
  (empty_plane_speed - speed_reduction_per_passenger * first_plane_passengers +
   empty_plane_speed - speed_reduction_per_passenger * second_plane_passengers +
   empty_plane_speed - speed_reduction_per_passenger * third_plane_passengers) / 3 = average_speed :=
by sorry

end third_plane_passenger_count_l2952_295213


namespace appropriate_methods_l2952_295278

/-- Represents different income levels of families -/
inductive IncomeLevel
  | High
  | Medium
  | Low

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Structure representing a survey -/
structure Survey where
  population : Nat
  sampleSize : Nat
  categories : Option (List (IncomeLevel × Nat))

/-- Function to determine the most appropriate sampling method for a given survey -/
def mostAppropriateMethod (s : Survey) : SamplingMethod :=
  if s.categories.isSome && s.population > 100 && s.sampleSize > 50
  then SamplingMethod.Stratified
  else SamplingMethod.SimpleRandom

/-- The two surveys from the problem -/
def survey1 : Survey :=
  { population := 420,
    sampleSize := 100,
    categories := some [(IncomeLevel.High, 125), (IncomeLevel.Medium, 200), (IncomeLevel.Low, 95)] }

def survey2 : Survey :=
  { population := 5,
    sampleSize := 3,
    categories := none }

/-- Theorem stating that the most appropriate methods for the given surveys are as expected -/
theorem appropriate_methods :
  (mostAppropriateMethod survey1 = SamplingMethod.Stratified) ∧
  (mostAppropriateMethod survey2 = SamplingMethod.SimpleRandom) := by
  sorry

end appropriate_methods_l2952_295278


namespace sum_of_products_bounds_l2952_295236

/-- Represents a table of -1s and 1s -/
def Table (n : ℕ) := Fin n → Fin n → Int

/-- Defines the valid entries for the table -/
def validEntry (x : Int) : Prop := x = 1 ∨ x = -1

/-- Defines a valid table where all entries are either 1 or -1 -/
def validTable (A : Table n) : Prop :=
  ∀ i j, validEntry (A i j)

/-- Product of elements in a row -/
def rowProduct (A : Table n) (i : Fin n) : Int :=
  (Finset.univ.prod fun j => A i j)

/-- Product of elements in a column -/
def colProduct (A : Table n) (j : Fin n) : Int :=
  (Finset.univ.prod fun i => A i j)

/-- Sum of products S for a given table -/
def sumOfProducts (A : Table n) : Int :=
  (Finset.univ.sum fun i => rowProduct A i) + (Finset.univ.sum fun j => colProduct A j)

/-- Theorem stating that the sum of products is always even and bounded -/
theorem sum_of_products_bounds (n : ℕ) (A : Table n) (h : validTable A) :
  ∃ k : Int, sumOfProducts A = 2 * k ∧ -n ≤ k ∧ k ≤ n :=
sorry

end sum_of_products_bounds_l2952_295236


namespace total_sum_is_correct_l2952_295211

/-- Represents the share ratios and total sum for a money division problem -/
structure MoneyDivision where
  a_ratio : ℝ
  b_ratio : ℝ
  c_ratio : ℝ
  c_share : ℝ
  total_sum : ℝ

/-- The money division problem with given ratios and c's share -/
def problem : MoneyDivision :=
  { a_ratio := 1
    b_ratio := 0.65
    c_ratio := 0.40
    c_share := 48
    total_sum := 246 }

/-- Theorem stating that the total sum is correct given the problem conditions -/
theorem total_sum_is_correct (p : MoneyDivision) :
  p.a_ratio = 1 ∧
  p.b_ratio = 0.65 ∧
  p.c_ratio = 0.40 ∧
  p.c_share = 48 →
  p.total_sum = 246 := by
  sorry

#check total_sum_is_correct problem

end total_sum_is_correct_l2952_295211


namespace history_book_cost_l2952_295205

theorem history_book_cost (total_books : ℕ) (math_book_cost : ℕ) (total_price : ℕ) (math_books : ℕ) :
  total_books = 80 →
  math_book_cost = 4 →
  total_price = 373 →
  math_books = 27 →
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 :=
by sorry

end history_book_cost_l2952_295205


namespace interest_rate_calculation_l2952_295279

theorem interest_rate_calculation (P R : ℝ) 
  (h1 : P * (1 + 4 * R / 100) = 400) 
  (h2 : P * (1 + 6 * R / 100) = 500) : 
  R = 25 := by
sorry

end interest_rate_calculation_l2952_295279


namespace symmetric_line_wrt_x_axis_l2952_295297

/-- Given a line with equation 3x - 4y + 5 = 0, its symmetric line with respect to the x-axis has the equation 3x + 4y + 5 = 0 -/
theorem symmetric_line_wrt_x_axis :
  ∀ (x y : ℝ), 3 * x - 4 * y + 5 = 0 →
  ∃ (x' y' : ℝ), x' = x ∧ y' = -y ∧ 3 * x' + 4 * y' + 5 = 0 :=
sorry

end symmetric_line_wrt_x_axis_l2952_295297


namespace number_of_triceratopses_l2952_295296

/-- Represents the number of rhinoceroses -/
def r : ℕ := sorry

/-- Represents the number of triceratopses -/
def t : ℕ := sorry

/-- The total number of horns -/
def total_horns : ℕ := 31

/-- The total number of legs -/
def total_legs : ℕ := 48

/-- Theorem stating that the number of triceratopses is 7 -/
theorem number_of_triceratopses : t = 7 := by
  sorry

end number_of_triceratopses_l2952_295296


namespace special_sequence_lower_bound_l2952_295245

/-- A sequence of n consecutive natural numbers with special divisor properties -/
structure SpecialSequence (n : ℕ) :=
  (original : Fin n → ℕ)
  (divisors : Fin n → ℕ)
  (original_ascending : ∀ i j, i < j → original i < original j)
  (divisors_ascending : ∀ i j, i < j → divisors i < divisors j)
  (divisor_property : ∀ i, 1 < divisors i ∧ divisors i < original i ∧ divisors i ∣ original i)

/-- All prime numbers smaller than n -/
def primes_less_than (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter Nat.Prime

/-- The main theorem -/
theorem special_sequence_lower_bound (n : ℕ) (seq : SpecialSequence n) :
  ∀ i, seq.original i > (n ^ (primes_less_than n).card) / (primes_less_than n).prod id :=
sorry

end special_sequence_lower_bound_l2952_295245


namespace emma_reaches_jack_emma_reaches_jack_proof_l2952_295250

/-- The time it takes for Emma to reach Jack given their initial conditions -/
theorem emma_reaches_jack : ℝ :=
  let initial_distance : ℝ := 30
  let combined_speed : ℝ := 2
  let jack_emma_speed_ratio : ℝ := 2
  let jack_stop_time : ℝ := 6
  
  33

theorem emma_reaches_jack_proof (initial_distance : ℝ) (combined_speed : ℝ) 
  (jack_emma_speed_ratio : ℝ) (jack_stop_time : ℝ) 
  (h1 : initial_distance = 30)
  (h2 : combined_speed = 2)
  (h3 : jack_emma_speed_ratio = 2)
  (h4 : jack_stop_time = 6) :
  emma_reaches_jack = 33 := by
  sorry

#check emma_reaches_jack_proof

end emma_reaches_jack_emma_reaches_jack_proof_l2952_295250


namespace min_value_fraction_l2952_295298

theorem min_value_fraction (x : ℝ) (h : x > 9) : 
  x^2 / (x - 9) ≥ 36 ∧ ∃ y > 9, y^2 / (y - 9) = 36 := by
sorry

end min_value_fraction_l2952_295298


namespace divisibility_by_power_of_five_l2952_295202

theorem divisibility_by_power_of_five :
  ∀ k : ℕ, ∃ n : ℕ, (5^k : ℕ) ∣ (n^2 + 1) := by
  sorry

end divisibility_by_power_of_five_l2952_295202


namespace special_function_uniqueness_l2952_295212

/-- A function satisfying the given properties -/
def special_function (g : ℝ → ℝ) : Prop :=
  g 2 = 2 ∧ ∀ x y : ℝ, g (x * y + g x) = x * g y + g x

/-- The main theorem stating that any function satisfying the special properties
    is equivalent to the function f(x) = 2x -/
theorem special_function_uniqueness (g : ℝ → ℝ) (hg : special_function g) :
  ∀ x : ℝ, g x = 2 * x :=
sorry

end special_function_uniqueness_l2952_295212


namespace tan_triangle_identity_l2952_295290

theorem tan_triangle_identity (A B C : Real) (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
  (h₄ : A + B + C = Real.pi) : 
  (Real.tan A * Real.tan B * Real.tan C) / (Real.tan A + Real.tan B + Real.tan C) = 1 :=
by sorry

end tan_triangle_identity_l2952_295290


namespace train_crossing_time_l2952_295295

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_length : ℝ) (signal_pole_time : ℝ) (platform_length : ℝ) :
  train_length = 300 →
  signal_pole_time = 24 →
  platform_length = 187.5 →
  (train_length + platform_length) / (train_length / signal_pole_time) = 39 :=
by sorry

end train_crossing_time_l2952_295295


namespace at_operation_difference_l2952_295225

def at_operation (x y : ℤ) : ℤ := x^2 * y - 3 * x

theorem at_operation_difference : at_operation 5 3 - at_operation 3 5 = 24 := by
  sorry

end at_operation_difference_l2952_295225


namespace workers_wage_problem_l2952_295218

/-- Worker's wage problem -/
theorem workers_wage_problem (total_days : ℕ) (overall_avg : ℝ) 
  (first_5_avg : ℝ) (second_5_avg : ℝ) (third_5_increase : ℝ) (last_5_decrease : ℝ) :
  total_days = 20 →
  overall_avg = 100 →
  first_5_avg = 90 →
  second_5_avg = 110 →
  third_5_increase = 0.05 →
  last_5_decrease = 0.10 →
  ∃ (eleventh_day_wage : ℝ),
    eleventh_day_wage = second_5_avg * (1 + third_5_increase) ∧
    eleventh_day_wage = 115.50 :=
by sorry

end workers_wage_problem_l2952_295218


namespace base_3_12021_equals_142_l2952_295230

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

theorem base_3_12021_equals_142 :
  base_3_to_10 [1, 2, 0, 2, 1] = 142 := by
  sorry

end base_3_12021_equals_142_l2952_295230


namespace final_ring_count_is_225_l2952_295239

/-- Calculates the final number of ornamental rings in the store after a series of transactions -/
def final_ring_count (initial_purchase : ℕ) (additional_purchase : ℕ) (final_sale : ℕ) : ℕ :=
  let initial_stock := initial_purchase / 2
  let total_stock := initial_purchase + initial_stock
  let remaining_after_first_sale := total_stock - (3 * total_stock / 4)
  let stock_after_additional_purchase := remaining_after_first_sale + additional_purchase
  stock_after_additional_purchase - final_sale

/-- The final number of ornamental rings in the store is 225 -/
theorem final_ring_count_is_225 : final_ring_count 200 300 150 = 225 := by
  sorry

end final_ring_count_is_225_l2952_295239


namespace vector_problem_l2952_295299

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

/-- Vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : Vector2D) : Prop := dot v w = 0

/-- The angle between two vectors is acute if their dot product is positive -/
def acuteAngle (v w : Vector2D) : Prop := dot v w > 0

theorem vector_problem (x : ℝ) : 
  let a : Vector2D := ⟨1, 2⟩
  let b : Vector2D := ⟨x, 1⟩
  (acuteAngle a b ↔ x > -2 ∧ x ≠ 1/2) ∧ 
  (perpendicular (Vector2D.mk (1 + 2*x) 4) (Vector2D.mk (2 - x) 3) ↔ x = 7/2) := by
  sorry

end vector_problem_l2952_295299


namespace union_of_A_and_B_l2952_295228

-- Define the sets A and B
def A (p : ℝ) : Set ℝ := {x | x^2 + p*x + 12 = 0}
def B (q : ℝ) : Set ℝ := {x | x^2 - 5*x + q = 0}

-- State the theorem
theorem union_of_A_and_B (p q : ℝ) :
  (Set.compl (A p) ∩ B q = {2}) →
  (A p ∩ Set.compl (B q) = {4}) →
  (A p ∪ B q = {2, 3, 6}) := by
  sorry


end union_of_A_and_B_l2952_295228


namespace insulation_minimum_cost_l2952_295294

/-- Represents the total cost function over 20 years for insulation thickness x (in cm) -/
def f (x : ℝ) : ℝ := 800 - 74 * x

/-- The domain of the function f -/
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 10

theorem insulation_minimum_cost :
  ∃ (x : ℝ), domain x ∧ f x = 700 ∧ ∀ (y : ℝ), domain y → f y ≥ f x :=
sorry

end insulation_minimum_cost_l2952_295294


namespace root_equation_value_l2952_295226

theorem root_equation_value (m : ℝ) : 
  m^2 + m - 1 = 0 → 3*m^2 + 3*m + 2006 = 2009 := by
  sorry

end root_equation_value_l2952_295226


namespace sin_2theta_value_l2952_295258

theorem sin_2theta_value (θ : Real) (h : Real.tan θ + 1 / Real.tan θ = 2) : 
  Real.sin (2 * θ) = 1 := by
  sorry

end sin_2theta_value_l2952_295258


namespace p_at_5_l2952_295243

/-- A monic quartic polynomial with specific values at x = 1, 2, 3, and 4 -/
def p : ℝ → ℝ :=
  fun x => x^4 + a*x^3 + b*x^2 + c*x + d
  where
    a : ℝ := sorry
    b : ℝ := sorry
    c : ℝ := sorry
    d : ℝ := sorry

/-- The polynomial p satisfies the given conditions -/
axiom p_cond1 : p 1 = 2
axiom p_cond2 : p 2 = 3
axiom p_cond3 : p 3 = 6
axiom p_cond4 : p 4 = 11

/-- The theorem to be proved -/
theorem p_at_5 : p 5 = 48 := by
  sorry

end p_at_5_l2952_295243


namespace wall_length_approximation_l2952_295268

/-- Given a square mirror and a rectangular wall, if the mirror's area is exactly half the wall's area,
    prove that the length of the wall is approximately 43 inches. -/
theorem wall_length_approximation (mirror_side : ℝ) (wall_width : ℝ) (wall_length : ℝ) : 
  mirror_side = 34 →
  wall_width = 54 →
  mirror_side ^ 2 = (wall_width * wall_length) / 2 →
  ∃ ε > 0, |wall_length - 43| < ε :=
by sorry

end wall_length_approximation_l2952_295268


namespace water_bucket_problem_l2952_295262

theorem water_bucket_problem (a b : ℝ) : 
  (a - 6 = (1/3) * (b + 6)) →
  (b - 6 = (1/2) * (a + 6)) →
  a = 13.2 := by
  sorry

end water_bucket_problem_l2952_295262


namespace top_square_after_folds_l2952_295266

/-- Represents a position on the 5x5 grid -/
structure Position :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents the state of the grid after folding -/
structure FoldedGrid :=
  (top_square : ℕ)
  (visible_squares : List ℕ)

/-- Initial numbering of the grid in row-major order -/
def initial_grid : Position → ℕ
  | ⟨r, c⟩ => r.val * 5 + c.val + 1

/-- Fold along the diagonal from bottom left to top right -/
def fold_diagonal (grid : Position → ℕ) : FoldedGrid :=
  sorry

/-- Fold the left half over the right half -/
def fold_left_to_right (fg : FoldedGrid) : FoldedGrid :=
  sorry

/-- Fold the top half over the bottom half -/
def fold_top_to_bottom (fg : FoldedGrid) : FoldedGrid :=
  sorry

/-- Fold the bottom half over the top half -/
def fold_bottom_to_top (fg : FoldedGrid) : FoldedGrid :=
  sorry

/-- Apply all folding steps -/
def apply_all_folds (grid : Position → ℕ) : FoldedGrid :=
  fold_bottom_to_top (fold_top_to_bottom (fold_left_to_right (fold_diagonal grid)))

theorem top_square_after_folds :
  (apply_all_folds initial_grid).top_square = 13 := by
  sorry

end top_square_after_folds_l2952_295266


namespace product_11_cubed_sum_l2952_295291

theorem product_11_cubed_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → a * b * c = 11^3 → a + b + c = 133 := by sorry

end product_11_cubed_sum_l2952_295291


namespace trader_gain_percentage_l2952_295252

/-- Represents a trader selling pens -/
structure PenTrader where
  sold : ℕ
  gainInPens : ℕ

/-- Calculates the gain percentage for a pen trader -/
def gainPercentage (trader : PenTrader) : ℚ :=
  (trader.gainInPens : ℚ) / (trader.sold : ℚ) * 100

/-- Theorem stating that for a trader selling 250 pens and gaining the cost of 65 pens, 
    the gain percentage is 26% -/
theorem trader_gain_percentage :
  ∀ (trader : PenTrader), 
    trader.sold = 250 → 
    trader.gainInPens = 65 → 
    gainPercentage trader = 26 := by
  sorry

end trader_gain_percentage_l2952_295252


namespace fraction_equality_implies_numerator_equality_l2952_295201

theorem fraction_equality_implies_numerator_equality
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b :=
by sorry

end fraction_equality_implies_numerator_equality_l2952_295201


namespace circular_arrangement_theorem_l2952_295257

/-- Represents a circular arrangement of people -/
structure CircularArrangement where
  n : ℕ  -- Total number of people
  dist : ℕ → ℕ → ℕ  -- Distance function between two positions

/-- The main theorem -/
theorem circular_arrangement_theorem (c : CircularArrangement) :
  (c.dist 31 7 = c.dist 31 14) → c.n = 41 := by
  sorry

/-- Helper function to calculate clockwise distance -/
def clockwise_distance (n : ℕ) (a b : ℕ) : ℕ :=
  if b ≥ a then b - a else n - a + b

/-- Axiom: The distance function in CircularArrangement is defined by clockwise_distance -/
axiom distance_defined (c : CircularArrangement) :
  ∀ a b, c.dist a b = clockwise_distance c.n a b

/-- Axiom: The arrangement is circular, so the distance from a to b equals the distance from b to a -/
axiom circular_symmetry (c : CircularArrangement) :
  ∀ a b, c.dist a b = c.dist b a

end circular_arrangement_theorem_l2952_295257


namespace tangent_line_x_intercept_l2952_295274

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 4*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 4

theorem tangent_line_x_intercept :
  let slope := f' 1
  let point := (1, f 1)
  let m := slope
  let b := point.2 - m * point.1
  (0 - b) / m = -3/7 :=
sorry

end tangent_line_x_intercept_l2952_295274


namespace school_event_handshakes_l2952_295261

/-- Represents the number of handshakes in a group of children -/
def handshakes (n : ℕ) : ℕ := 
  (n * (n - 1)) / 2

/-- The problem statement -/
theorem school_event_handshakes : 
  handshakes 8 = 36 := by sorry

end school_event_handshakes_l2952_295261


namespace minimum_weights_l2952_295241

def is_valid_weight_set (weights : List ℕ) : Prop :=
  ∀ w : ℕ, 1 ≤ w ∧ w ≤ 20 →
    ∃ (a b : ℕ), a ∈ weights ∧ b ∈ weights ∧ (w = a ∨ w = a + b)

theorem minimum_weights :
  ∃ (weights : List ℕ),
    weights.length = 6 ∧
    is_valid_weight_set weights ∧
    ∀ (other_weights : List ℕ),
      is_valid_weight_set other_weights →
      other_weights.length ≥ 6 :=
sorry

end minimum_weights_l2952_295241


namespace max_grandchildren_problem_l2952_295272

/-- The number of children Max has -/
def max_children : ℕ := 8

/-- The number of Max's children who have the same number of children as Max -/
def children_with_same : ℕ := 6

/-- The total number of Max's grandchildren -/
def total_grandchildren : ℕ := 58

/-- The number of children each exception has -/
def exception_children : ℕ := 5

theorem max_grandchildren_problem :
  (children_with_same * max_children) + 
  (2 * exception_children) = total_grandchildren :=
by sorry

end max_grandchildren_problem_l2952_295272


namespace opposite_of_23_l2952_295270

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_23 : opposite 23 = -23 := by
  sorry

end opposite_of_23_l2952_295270


namespace kit_savings_percentage_l2952_295265

/-- The price of the camera lens filter kit -/
def kit_price : ℚ := 75.50

/-- The number of filters in the kit -/
def num_filters : ℕ := 5

/-- The price of the first type of filter -/
def filter_price1 : ℚ := 7.35

/-- The number of filters of the first type -/
def num_filters1 : ℕ := 3

/-- The price of the second type of filter -/
def filter_price2 : ℚ := 12.05

/-- The number of filters of the second type (only 2 are used in the kit) -/
def num_filters2 : ℕ := 2

/-- The price of the third type of filter -/
def filter_price3 : ℚ := 12.50

/-- The number of filters of the third type -/
def num_filters3 : ℕ := 1

/-- The total price of filters if purchased individually -/
def total_individual_price : ℚ :=
  filter_price1 * num_filters1 + filter_price2 * num_filters2 + filter_price3 * num_filters3

/-- The amount saved by purchasing the kit -/
def amount_saved : ℚ := total_individual_price - kit_price

/-- The percentage saved by purchasing the kit -/
def percentage_saved : ℚ := (amount_saved / total_individual_price) * 100

theorem kit_savings_percentage :
  percentage_saved = 28.72 := by sorry

end kit_savings_percentage_l2952_295265
