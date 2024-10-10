import Mathlib

namespace a_properties_l2822_282222

def a (n : ℕ+) : ℚ := (n - 1) / n

theorem a_properties :
  (∀ n : ℕ+, a n < 1) ∧
  (∀ n : ℕ+, a (n + 1) > a n) :=
by sorry

end a_properties_l2822_282222


namespace randy_piano_expertise_l2822_282230

/-- Represents the number of days in a year --/
def daysPerYear : ℕ := 365

/-- Represents the number of weeks in a year --/
def weeksPerYear : ℕ := 52

/-- Represents Randy's current age --/
def currentAge : ℕ := 12

/-- Represents Randy's target age to become an expert --/
def targetAge : ℕ := 20

/-- Represents the number of practice days per week --/
def practiceDaysPerWeek : ℕ := 5

/-- Represents the number of practice hours per day --/
def practiceHoursPerDay : ℕ := 5

/-- Represents the total hours needed to become an expert --/
def expertiseHours : ℕ := 10000

/-- Theorem stating that Randy can take 10 days of vacation per year and still achieve expertise --/
theorem randy_piano_expertise :
  ∃ (vacationDaysPerYear : ℕ),
    vacationDaysPerYear = 10 ∧
    (targetAge - currentAge) * weeksPerYear * practiceDaysPerWeek * practiceHoursPerDay -
    (targetAge - currentAge) * vacationDaysPerYear * practiceHoursPerDay ≥ expertiseHours :=
by sorry

end randy_piano_expertise_l2822_282230


namespace least_sum_m_n_l2822_282270

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (Nat.gcd (m.val + n.val) 330 = 1) ∧ 
  (∃ (k : ℕ), m.val ^ m.val = k * (n.val ^ n.val)) ∧ 
  (∀ (l : ℕ), m.val ≠ l * n.val) ∧
  (m.val + n.val = 390) ∧
  (∀ (p q : ℕ+), 
    (Nat.gcd (p.val + q.val) 330 = 1) → 
    (∃ (k : ℕ), p.val ^ p.val = k * (q.val ^ q.val)) → 
    (∀ (l : ℕ), p.val ≠ l * q.val) → 
    (p.val + q.val ≥ 390)) := by
  sorry

end least_sum_m_n_l2822_282270


namespace compound_molar_mass_l2822_282218

/-- Given a compound where 6 moles weighs 612 grams, prove its molar mass is 102 grams per mole -/
theorem compound_molar_mass (mass : ℝ) (moles : ℝ) (h1 : mass = 612) (h2 : moles = 6) :
  mass / moles = 102 := by
  sorry

end compound_molar_mass_l2822_282218


namespace division_problem_l2822_282299

theorem division_problem : 180 / (12 + 13 * 2) = 45 / 19 := by
  sorry

end division_problem_l2822_282299


namespace rectangular_prism_width_l2822_282246

theorem rectangular_prism_width 
  (l h d : ℝ) 
  (hl : l = 6) 
  (hh : h = 8) 
  (hd : d = 15) 
  (h_diagonal : d^2 = l^2 + w^2 + h^2) : 
  w = 5 * Real.sqrt 5 :=
sorry

end rectangular_prism_width_l2822_282246


namespace lower_bound_sum_squares_roots_l2822_282290

/-- A monic polynomial of degree 4 with real coefficients -/
structure MonicPolynomial4 where
  coeffs : Fin 4 → ℝ
  monic : coeffs 0 = 1

/-- The sum of the squares of the roots of a polynomial -/
def sumSquaresRoots (p : MonicPolynomial4) : ℝ := sorry

/-- The theorem statement -/
theorem lower_bound_sum_squares_roots (p : MonicPolynomial4)
  (h1 : p.coeffs 1 = 0)  -- No cubic term
  (h2 : ∃ a₂ : ℝ, p.coeffs 2 = a₂ ∧ p.coeffs 3 = 2 * a₂) :  -- a₃ = 2a₂
  |sumSquaresRoots p| ≥ (1/4 : ℝ) := by sorry

end lower_bound_sum_squares_roots_l2822_282290


namespace least_positive_integer_with_remainders_l2822_282225

theorem least_positive_integer_with_remainders : ∃! a : ℕ,
  a > 0 ∧
  a % 2 = 1 ∧
  a % 3 = 2 ∧
  a % 4 = 3 ∧
  a % 5 = 4 ∧
  ∀ b : ℕ, b > 0 ∧ b % 2 = 1 ∧ b % 3 = 2 ∧ b % 4 = 3 ∧ b % 5 = 4 → a ≤ b :=
by
  use 59
  sorry

end least_positive_integer_with_remainders_l2822_282225


namespace accounting_balance_l2822_282296

/-- Given the equation 3q - x = 15000, where q = 7 and x = 7 + 75i, prove that p = 5005 + 25i -/
theorem accounting_balance (q x p : ℂ) : 
  3 * q - x = 15000 → q = 7 → x = 7 + 75 * Complex.I → p = 5005 + 25 * Complex.I := by
  sorry

end accounting_balance_l2822_282296


namespace contrapositive_odd_product_l2822_282272

theorem contrapositive_odd_product (a b : ℤ) :
  (¬(Odd (a * b)) → ¬(Odd a ∧ Odd b)) ↔
  ((Odd a ∧ Odd b) → Odd (a * b)) :=
sorry

end contrapositive_odd_product_l2822_282272


namespace cucumber_weight_problem_l2822_282205

theorem cucumber_weight_problem (initial_water_percentage : Real)
                                (final_water_percentage : Real)
                                (final_weight : Real) :
  initial_water_percentage = 0.99 →
  final_water_percentage = 0.95 →
  final_weight = 20 →
  ∃ initial_weight : Real,
    initial_weight = 100 ∧
    (1 - initial_water_percentage) * initial_weight =
    (1 - final_water_percentage) * final_weight :=
by sorry

end cucumber_weight_problem_l2822_282205


namespace tan_two_implications_l2822_282266

theorem tan_two_implications (θ : Real) (h : Real.tan θ = 2) : 
  (Real.cos θ)^2 = 1/5 ∧ (Real.sin θ)^2 = 4/5 ∧ 
  (4 * Real.sin θ - 3 * Real.cos θ) / (6 * Real.cos θ + 2 * Real.sin θ) = 1/2 := by
  sorry

end tan_two_implications_l2822_282266


namespace sticker_distribution_count_l2822_282291

/-- The number of ways to partition n identical objects into k or fewer parts -/
def partition_count (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of stickers -/
def num_stickers : ℕ := 9

/-- The number of sheets -/
def num_sheets : ℕ := 3

theorem sticker_distribution_count : 
  partition_count num_stickers num_sheets = 12 := by sorry

end sticker_distribution_count_l2822_282291


namespace min_volume_ratio_l2822_282209

/-- A spherical cap -/
structure SphericalCap where
  volume : ℝ

/-- A cylinder -/
structure Cylinder where
  volume : ℝ

/-- Configuration of a spherical cap and cylinder sharing a common inscribed sphere -/
structure Configuration where
  cap : SphericalCap
  cylinder : Cylinder
  bottom_faces_on_same_plane : Prop
  share_common_inscribed_sphere : Prop

/-- The minimum volume ratio theorem -/
theorem min_volume_ratio (config : Configuration) :
  ∃ (min_ratio : ℝ), min_ratio = 4/3 ∧
  ∀ (ratio : ℝ), ratio = config.cap.volume / config.cylinder.volume → min_ratio ≤ ratio :=
sorry

end min_volume_ratio_l2822_282209


namespace complex_equation_solution_l2822_282285

theorem complex_equation_solution (z : ℂ) :
  (Complex.I * 3 + Real.sqrt 3) * z = Complex.I * 3 →
  z = 3 / 4 + Complex.I * (Real.sqrt 3 / 4) := by
  sorry

end complex_equation_solution_l2822_282285


namespace cos_negative_75_degrees_l2822_282243

theorem cos_negative_75_degrees :
  Real.cos (-(75 * π / 180)) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_negative_75_degrees_l2822_282243


namespace range_of_f_when_a_is_2_properties_of_M_l2822_282216

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - a*x + 4 - a^2

-- Theorem for the range of f when a = 2
theorem range_of_f_when_a_is_2 :
  ∃ (y : ℝ), y ∈ Set.Icc (-1) 8 ↔ ∃ (x : ℝ), x ∈ Set.Icc (-2) 3 ∧ f 2 x = y :=
sorry

-- Define the set M
def M : Set ℝ := {4}

-- Theorem for the properties of M
theorem properties_of_M :
  (4 ∈ M) ∧
  (∀ a ∈ M, ∀ x ∈ Set.Icc (-2) 2, f a x ≤ 0) ∧
  (∃ b ∉ M, ∀ x ∈ Set.Icc (-2) 2, f b x ≤ 0) :=
sorry

end range_of_f_when_a_is_2_properties_of_M_l2822_282216


namespace simplify_fraction_l2822_282207

theorem simplify_fraction : (210 : ℚ) / 7350 * 14 = 2 / 5 := by sorry

end simplify_fraction_l2822_282207


namespace lemon_candy_count_l2822_282221

theorem lemon_candy_count (total : ℕ) (caramel : ℕ) (p : ℚ) (lemon : ℕ) : 
  caramel = 3 →
  p = 3 / 7 →
  p = caramel / total →
  lemon = total - caramel →
  lemon = 4 := by
sorry

end lemon_candy_count_l2822_282221


namespace amanda_keeps_33_candy_bars_l2822_282293

/-- Calculates the number of candy bars Amanda keeps for herself after a series of events --/
def amanda_candy_bars : ℕ :=
  let initial := 7
  let after_first_give := initial - (initial / 3)
  let after_buying := after_first_give + 30
  let after_second_give := after_buying - (after_buying / 4)
  let after_gift := after_second_give + 15
  let final := after_gift - ((15 * 3) / 5)
  final

/-- Theorem stating that Amanda keeps 33 candy bars for herself --/
theorem amanda_keeps_33_candy_bars : amanda_candy_bars = 33 := by
  sorry

end amanda_keeps_33_candy_bars_l2822_282293


namespace hamburgers_served_l2822_282210

theorem hamburgers_served (total : Nat) (leftover : Nat) (served : Nat) :
  total = 9 → leftover = 6 → served = total - leftover → served = 3 := by
  sorry

end hamburgers_served_l2822_282210


namespace reciprocal_sum_property_l2822_282271

theorem reciprocal_sum_property (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : 1 / a + 1 / b + 1 / c = 1 / (a + b + c)) :
  ∀ n : ℤ, (1 / a^n + 1 / b^n + 1 / c^n = 1 / (a^n + b^n + c^n)) ↔ ∃ k : ℕ, n = 2 * k - 1 := by
  sorry

end reciprocal_sum_property_l2822_282271


namespace number_of_green_balls_l2822_282281

/-- Given a bag with blue and green balls, prove the number of green balls -/
theorem number_of_green_balls
  (blue_balls : ℕ)
  (total_balls : ℕ)
  (h_blue_balls : blue_balls = 9)
  (h_prob_blue : blue_balls / total_balls = 3 / 10)
  (h_total : total_balls = blue_balls + green_balls)
  (green_balls : ℕ) :
  green_balls = 21 := by
  sorry

#check number_of_green_balls

end number_of_green_balls_l2822_282281


namespace jenny_easter_eggs_l2822_282248

theorem jenny_easter_eggs (n : ℕ) : 
  n ∣ 30 ∧ n ∣ 45 ∧ n ≥ 5 → n ≤ 15 :=
by sorry

end jenny_easter_eggs_l2822_282248


namespace world_cup_gifts_l2822_282288

/-- Calculates the number of gifts needed for a world cup inauguration event. -/
def gifts_needed (num_teams : ℕ) : ℕ :=
  num_teams * 2

/-- Theorem: The number of gifts needed for the world cup inauguration event with 7 teams is 14. -/
theorem world_cup_gifts : gifts_needed 7 = 14 := by
  sorry

end world_cup_gifts_l2822_282288


namespace potatoes_for_mashed_l2822_282203

theorem potatoes_for_mashed (initial : ℕ) (salad : ℕ) (remaining : ℕ) : 
  initial = 52 → salad = 15 → remaining = 13 → initial - salad - remaining = 24 := by
  sorry

end potatoes_for_mashed_l2822_282203


namespace f_values_f_inequality_range_l2822_282238

noncomputable section

variable (f : ℝ → ℝ)

axiom domain : ∀ x, x > 0 → f x ≠ 0
axiom f_2 : f 2 = 1
axiom f_mult : ∀ x y, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_increasing : ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂

theorem f_values :
  f 1 = 0 ∧ f 4 = 2 ∧ f 8 = 3 :=
sorry

theorem f_inequality_range :
  ∀ x, (f x + f (x - 2) ≤ 3) ↔ (2 < x ∧ x ≤ 4) :=
sorry

end f_values_f_inequality_range_l2822_282238


namespace f_max_value_l2822_282267

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := (S n) / ((n + 32) * (S (n + 1)))

theorem f_max_value : ∀ n : ℕ, f n ≤ 1 / 50 := by sorry

end f_max_value_l2822_282267


namespace carbonic_acid_weight_is_62_024_l2822_282250

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.011

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 15.999

/-- The molecular formula of carbonic acid -/
structure CarbenicAcid where
  hydrogen : ℕ := 2
  carbon : ℕ := 1
  oxygen : ℕ := 3

/-- The molecular weight of carbonic acid in atomic mass units (amu) -/
def carbonic_acid_weight (acid : CarbenicAcid) : ℝ :=
  acid.hydrogen * hydrogen_weight + 
  acid.carbon * carbon_weight + 
  acid.oxygen * oxygen_weight

/-- Theorem stating that the molecular weight of carbonic acid is 62.024 amu -/
theorem carbonic_acid_weight_is_62_024 :
  carbonic_acid_weight { } = 62.024 := by
  sorry

end carbonic_acid_weight_is_62_024_l2822_282250


namespace sixteen_even_numbers_l2822_282269

/-- Represents a card with two numbers -/
structure Card where
  front : Nat
  back : Nat

/-- Counts the number of three-digit even numbers that can be formed from the given cards -/
def countEvenNumbers (cards : List Card) : Nat :=
  cards.foldl (fun acc card => 
    acc + (if card.front % 2 == 0 then 1 else 0) + 
          (if card.back % 2 == 0 then 1 else 0)
  ) 0

/-- The main theorem stating that 16 different three-digit even numbers can be formed -/
theorem sixteen_even_numbers : 
  let cards := [Card.mk 0 1, Card.mk 2 3, Card.mk 4 5]
  countEvenNumbers cards = 16 := by
  sorry


end sixteen_even_numbers_l2822_282269


namespace unpainted_cubes_count_l2822_282212

/-- Represents a 5x5x5 cube with painted faces -/
structure PaintedCube where
  size : Nat
  painted_squares_per_face : Nat
  total_cubes : Nat
  painted_pattern_size : Nat

/-- Calculates the number of unpainted cubes in the PaintedCube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  cube.total_cubes - (cube.painted_squares_per_face * 6 - (cube.painted_pattern_size - 1) * 4 * 3)

/-- Theorem stating that the number of unpainted cubes is 83 -/
theorem unpainted_cubes_count (cube : PaintedCube) 
  (h1 : cube.size = 5)
  (h2 : cube.painted_squares_per_face = 9)
  (h3 : cube.total_cubes = 125)
  (h4 : cube.painted_pattern_size = 3) : 
  unpainted_cubes cube = 83 := by
  sorry

#eval unpainted_cubes { size := 5, painted_squares_per_face := 9, total_cubes := 125, painted_pattern_size := 3 }

end unpainted_cubes_count_l2822_282212


namespace f_properties_l2822_282231

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.log x + a) / x

def monotonicity_intervals (a : ℝ) : Prop :=
  (a > 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp (1 - a) → f a x₁ < f a x₂) ∧
            (∀ x₁ x₂, Real.exp (1 - a) < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂)) ∧
  (a < 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp (1 - a) → f a x₁ > f a x₂) ∧
            (∀ x₁ x₂, Real.exp (1 - a) < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂))

def has_root_in_interval (a : ℝ) : Prop :=
  ∃ x, Real.exp 1 < x ∧ f a x = 0

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  monotonicity_intervals a ∧ (has_root_in_interval a ↔ a < -1) :=
sorry

end

end f_properties_l2822_282231


namespace fourth_term_coefficient_binomial_expansion_l2822_282287

theorem fourth_term_coefficient_binomial_expansion :
  let n : ℕ := 5
  let a : ℤ := 2
  let b : ℤ := -3
  let k : ℕ := 3  -- For the fourth term, we choose 3 from 5
  (n.choose k) * a^(n - k) * b^k = 720 :=
by sorry

end fourth_term_coefficient_binomial_expansion_l2822_282287


namespace xiaojun_original_money_l2822_282224

/-- The amount of money Xiaojun originally had -/
def original_money : ℝ := 30

/-- The daily allowance Xiaojun receives from his dad -/
def daily_allowance : ℝ := 5

/-- The number of days Xiaojun can last when spending 10 yuan per day -/
def days_at_10 : ℝ := 6

/-- The number of days Xiaojun can last when spending 15 yuan per day -/
def days_at_15 : ℝ := 3

/-- The daily spending when Xiaojun lasts for 6 days -/
def spending_10 : ℝ := 10

/-- The daily spending when Xiaojun lasts for 3 days -/
def spending_15 : ℝ := 15

theorem xiaojun_original_money :
  (days_at_10 * spending_10 - days_at_10 * daily_allowance = original_money) ∧
  (days_at_15 * spending_15 - days_at_15 * daily_allowance = original_money) :=
by sorry

end xiaojun_original_money_l2822_282224


namespace floor_equation_equivalence_l2822_282289

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The solution set for the equation -/
def solution_set : Set ℝ :=
  {x | x < 0 ∨ x ≥ 2.5}

/-- Theorem stating the equivalence of the equation and the solution set -/
theorem floor_equation_equivalence (x : ℝ) :
  floor (1 / (1 - x)) = floor (1 / (1.5 - x)) ↔ x ∈ solution_set :=
sorry

end floor_equation_equivalence_l2822_282289


namespace max_points_in_configuration_l2822_282294

/-- A configuration of points in the plane with associated real numbers -/
structure PointConfiguration where
  n : ℕ
  points : Fin n → ℝ × ℝ
  radii : Fin n → ℝ
  distance_property : ∀ (i j : Fin n), i ≠ j →
    Real.sqrt ((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2 = radii i + radii j

/-- The maximum number of points in a valid configuration is 4 -/
theorem max_points_in_configuration :
  (∃ (c : PointConfiguration), c.n = 4) ∧
  (∀ (c : PointConfiguration), c.n ≤ 4) :=
sorry

end max_points_in_configuration_l2822_282294


namespace binomial_square_constant_l2822_282259

/-- If 4x^2 + 12x + a is the square of a binomial, then a = 9 -/
theorem binomial_square_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 4*x^2 + 12*x + a = (2*x + b)^2) → a = 9 := by
  sorry

end binomial_square_constant_l2822_282259


namespace ellipse_foci_distance_l2822_282244

/-- The distance between foci of an ellipse with given parameters -/
theorem ellipse_foci_distance (a b : ℝ) (h1 : a = 10) (h2 : b = 8) (h3 : a > b) :
  2 * Real.sqrt (a^2 - b^2) = 12 := by
  sorry

#check ellipse_foci_distance

end ellipse_foci_distance_l2822_282244


namespace contrapositive_truth_l2822_282233

/-- The function f(x) = x^2 - mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 1

/-- The theorem statement -/
theorem contrapositive_truth (m : ℝ) 
  (h : ∀ x > 0, f m x ≥ 0) :
  ∀ a b, a > 0 → b > 0 → 
    (a + b ≤ 1 → 1/a + 2/b ≥ 3 + Real.sqrt 2 * m) :=
by sorry

end contrapositive_truth_l2822_282233


namespace fraction_product_simplification_l2822_282265

theorem fraction_product_simplification :
  (2 / 3) * (3 / 7) * (7 / 4) * (4 / 5) * (5 / 6) = 1 / 3 := by
  sorry

end fraction_product_simplification_l2822_282265


namespace distance_P_to_y_axis_l2822_282241

/-- The distance from a point to the y-axis in a Cartesian coordinate system --/
def distance_to_y_axis (x y : ℝ) : ℝ := |x|

/-- The point P --/
def P : ℝ × ℝ := (-3, 4)

/-- Theorem: The distance from P(-3, 4) to the y-axis is 3 --/
theorem distance_P_to_y_axis :
  distance_to_y_axis P.1 P.2 = 3 := by
  sorry


end distance_P_to_y_axis_l2822_282241


namespace percentage_not_covering_politics_l2822_282249

-- Define the percentages as real numbers
def country_x : ℝ := 15
def country_y : ℝ := 10
def country_z : ℝ := 8
def x_elections : ℝ := 6
def y_foreign : ℝ := 5
def z_social : ℝ := 3
def not_local : ℝ := 50
def international : ℝ := 5
def economics : ℝ := 2

-- Theorem statement
theorem percentage_not_covering_politics :
  100 - (country_x + country_y + country_z + international + economics + not_local) = 10 := by
  sorry

end percentage_not_covering_politics_l2822_282249


namespace parabola_focus_line_l2822_282286

/-- Given a parabola and a line passing through its focus, prove the value of p -/
theorem parabola_focus_line (p : ℝ) (A B : ℝ × ℝ) : 
  p > 0 →  -- p is positive
  (∀ x y, y = x^2 / (2*p)) →  -- equation of parabola
  (A.1^2 = 2*p*A.2) →  -- A is on the parabola
  (B.1^2 = 2*p*B.2) →  -- B is on the parabola
  (A.1 + B.1 = 2) →  -- midpoint of AB has x-coordinate 1
  ((A.2 + B.2) / 2 = 1) →  -- midpoint of AB has y-coordinate 1
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36 →  -- length of AB is 6
  p = 2 := by
sorry

end parabola_focus_line_l2822_282286


namespace twice_square_sum_l2822_282236

theorem twice_square_sum (x y : ℤ) : x^4 + y^4 + (x+y)^4 = 2 * (x^2 + x*y + y^2)^2 := by
  sorry

end twice_square_sum_l2822_282236


namespace logarithm_equation_l2822_282284

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_equation : log10 5 * log10 50 - log10 2 * log10 20 - log10 625 = -2 := by
  sorry

end logarithm_equation_l2822_282284


namespace sum_of_digits_of_product_l2822_282297

/-- Represents a number formed by repeating a pattern a certain number of times -/
def repeatedPattern (pattern : ℕ) (repetitions : ℕ) : ℕ :=
  sorry

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem sum_of_digits_of_product : 
  let a := repeatedPattern 15 1004 * repeatedPattern 3 52008
  sumOfDigits a = 18072 :=
sorry

end sum_of_digits_of_product_l2822_282297


namespace apple_cost_calculation_l2822_282298

/-- Calculates the total cost of apples for Irene and her dog for 2 weeks -/
def apple_cost (apple_weight : Real) (red_price : Real) (green_price : Real) 
  (red_increase : Real) (green_decrease : Real) : Real :=
  let apples_needed := 14 * 0.5
  let pounds_needed := apples_needed * apple_weight
  let week1_cost := pounds_needed * red_price
  let week2_cost := pounds_needed * (green_price * (1 - green_decrease))
  week1_cost + week2_cost

theorem apple_cost_calculation :
  apple_cost (1/4) 2 2.5 0.1 0.05 = 7.65625 := by
  sorry

end apple_cost_calculation_l2822_282298


namespace coke_calories_is_215_l2822_282282

/-- Represents the calorie content of various food items and meals --/
structure CalorieContent where
  cake : ℕ
  chips : ℕ
  breakfast : ℕ
  lunch : ℕ
  dailyLimit : ℕ
  remainingAfterCoke : ℕ

/-- Calculates the calorie content of the coke --/
def cokeCalories (c : CalorieContent) : ℕ :=
  c.dailyLimit - (c.cake + c.chips + c.breakfast + c.lunch) - c.remainingAfterCoke

/-- Theorem stating that the coke has 215 calories --/
theorem coke_calories_is_215 (c : CalorieContent) 
  (h1 : c.cake = 110)
  (h2 : c.chips = 310)
  (h3 : c.breakfast = 560)
  (h4 : c.lunch = 780)
  (h5 : c.dailyLimit = 2500)
  (h6 : c.remainingAfterCoke = 525) :
  cokeCalories c = 215 := by
  sorry

#eval cokeCalories { cake := 110, chips := 310, breakfast := 560, lunch := 780, dailyLimit := 2500, remainingAfterCoke := 525 }

end coke_calories_is_215_l2822_282282


namespace bob_first_six_probability_l2822_282295

/-- The probability of tossing a six on a fair die -/
def probSix : ℚ := 1 / 6

/-- The probability of not tossing a six on a fair die -/
def probNotSix : ℚ := 1 - probSix

/-- The order of players: Alice, Charlie, Bob -/
inductive Player : Type
| Alice : Player
| Charlie : Player
| Bob : Player

/-- The probability that Bob is the first to toss a six in the die-tossing game -/
def probBobFirstSix : ℚ := 25 / 91

theorem bob_first_six_probability :
  probBobFirstSix = (probNotSix * probNotSix * probSix) / (1 - probNotSix * probNotSix * probNotSix) :=
by sorry

end bob_first_six_probability_l2822_282295


namespace base9_addition_l2822_282275

-- Define a function to convert a base 9 number to base 10
def base9ToBase10 (n : List Nat) : Nat :=
  n.foldr (fun digit acc => acc * 9 + digit) 0

-- Define a function to convert a base 10 number to base 9
def base10ToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

-- Define the numbers in base 9
def a : List Nat := [2, 5, 4]
def b : List Nat := [6, 2, 7]
def c : List Nat := [5, 0, 3]

-- Define the expected result in base 9
def result : List Nat := [1, 4, 8, 5]

theorem base9_addition :
  base10ToBase9 (base9ToBase10 a + base9ToBase10 b + base9ToBase10 c) = result :=
sorry

end base9_addition_l2822_282275


namespace max_axes_of_symmetry_l2822_282257

/-- A type representing a configuration of segments on a plane -/
structure SegmentConfiguration where
  k : ℕ+  -- number of segments (positive natural number)

/-- The number of axes of symmetry for a given segment configuration -/
def axesOfSymmetry (config : SegmentConfiguration) : ℕ := sorry

/-- Theorem stating that the maximum number of axes of symmetry is 2k -/
theorem max_axes_of_symmetry (config : SegmentConfiguration) :
  ∃ (arrangement : SegmentConfiguration), 
    arrangement.k = config.k ∧ 
    axesOfSymmetry arrangement = 2 * config.k.val ∧
    ∀ (other : SegmentConfiguration), 
      other.k = config.k → 
      axesOfSymmetry other ≤ axesOfSymmetry arrangement :=
by sorry

end max_axes_of_symmetry_l2822_282257


namespace chocolate_vanilla_survey_l2822_282262

theorem chocolate_vanilla_survey (total : ℕ) (chocolate : ℕ) (vanilla : ℕ) 
  (h_total : total = 120)
  (h_chocolate : chocolate = 95)
  (h_vanilla : vanilla = 85) :
  (chocolate + vanilla - total : ℕ) ≥ 25 :=
by sorry

end chocolate_vanilla_survey_l2822_282262


namespace quadratic_equation_with_prime_coefficients_l2822_282204

theorem quadratic_equation_with_prime_coefficients (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ x : ℤ, x^2 - p*x + q = 0) → p = 3 :=
by sorry

end quadratic_equation_with_prime_coefficients_l2822_282204


namespace cost_sharing_ratio_l2822_282201

def monthly_cost : ℚ := 14
def your_payment : ℚ := 84
def total_months : ℕ := 12

theorem cost_sharing_ratio :
  let yearly_cost := monthly_cost * total_months
  let friend_payment := yearly_cost - your_payment
  your_payment = friend_payment :=
by sorry

end cost_sharing_ratio_l2822_282201


namespace three_number_problem_l2822_282215

theorem three_number_problem (x y z : ℝ) 
  (sum_eq : x + y + z = 500)
  (x_eq : x = 200)
  (y_eq : y = 2 * z)
  (diff_eq : x - z = 0.5 * y) :
  z = 100 := by
  sorry

end three_number_problem_l2822_282215


namespace pure_imaginary_z_l2822_282292

theorem pure_imaginary_z (a : ℝ) : 
  let z : ℂ := a^2 + 2*a - 2 + (2*Complex.I)/(1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 ∨ a = -3 :=
by sorry

end pure_imaginary_z_l2822_282292


namespace percentage_proof_l2822_282280

/-- Given a number N and a percentage P, proves that P is 50% when N is 456 and P% of N equals 40% of 120 plus 180. -/
theorem percentage_proof (N : ℝ) (P : ℝ) : 
  N = 456 →
  (P / 100) * N = (40 / 100) * 120 + 180 →
  P = 50 := by
sorry

end percentage_proof_l2822_282280


namespace maria_cookies_distribution_l2822_282235

/-- Calculates the number of cookies per bag given the total number of cookies and the number of bags. -/
def cookiesPerBag (totalCookies : ℕ) (numBags : ℕ) : ℕ :=
  totalCookies / numBags

theorem maria_cookies_distribution (chocolateChipCookies oatmealCookies numBags : ℕ) 
  (h1 : chocolateChipCookies = 33)
  (h2 : oatmealCookies = 2)
  (h3 : numBags = 7) :
  cookiesPerBag (chocolateChipCookies + oatmealCookies) numBags = 5 := by
  sorry

end maria_cookies_distribution_l2822_282235


namespace pyramid_faces_l2822_282239

/-- A polygonal pyramid with a regular polygon base -/
structure PolygonalPyramid where
  base_sides : ℕ
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Properties of the polygonal pyramid -/
def pyramid_properties (p : PolygonalPyramid) : Prop :=
  p.vertices = p.base_sides + 1 ∧
  p.edges = 2 * p.base_sides ∧
  p.faces = p.base_sides + 1 ∧
  p.edges + p.vertices = 1915

theorem pyramid_faces (p : PolygonalPyramid) (h : pyramid_properties p) : p.faces = 639 := by
  sorry

end pyramid_faces_l2822_282239


namespace polar_midpoint_specific_case_l2822_282234

/-- The midpoint of a line segment in polar coordinates --/
def polar_midpoint (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

theorem polar_midpoint_specific_case :
  let (r, θ) := polar_midpoint 10 (π/3) 10 (5*π/6)
  r = 5 * Real.sqrt 2 ∧ θ = 2*π/3 :=
sorry

end polar_midpoint_specific_case_l2822_282234


namespace apple_sales_remaining_fraction_l2822_282273

/-- Proves that the fraction of money remaining after repairs is 1/5 --/
theorem apple_sales_remaining_fraction (apple_price : ℚ) (bike_cost : ℚ) (repair_percentage : ℚ) (apples_sold : ℕ) :
  apple_price = 5/4 →
  bike_cost = 80 →
  repair_percentage = 1/4 →
  apples_sold = 20 →
  let total_earnings := apple_price * apples_sold
  let repair_cost := repair_percentage * bike_cost
  let remaining := total_earnings - repair_cost
  remaining / total_earnings = 1/5 := by
  sorry


end apple_sales_remaining_fraction_l2822_282273


namespace no_same_color_neighbors_probability_l2822_282251

-- Define the number of beads for each color
def num_red : Nat := 5
def num_white : Nat := 3
def num_blue : Nat := 2

-- Define the total number of beads
def total_beads : Nat := num_red + num_white + num_blue

-- Define a function to calculate the number of valid arrangements
def valid_arrangements : Nat := 0

-- Define a function to calculate the total number of possible arrangements
def total_arrangements : Nat := Nat.factorial total_beads / (Nat.factorial num_red * Nat.factorial num_white * Nat.factorial num_blue)

-- Theorem: The probability of no two neighboring beads being the same color is 0
theorem no_same_color_neighbors_probability :
  (valid_arrangements : ℚ) / total_arrangements = 0 := by sorry

end no_same_color_neighbors_probability_l2822_282251


namespace sunflower_seed_distribution_l2822_282283

theorem sunflower_seed_distribution (total_seeds : ℕ) (num_cans : ℕ) (seeds_per_can : ℕ) 
  (h1 : total_seeds = 54)
  (h2 : num_cans = 9)
  (h3 : total_seeds = num_cans * seeds_per_can) :
  seeds_per_can = 6 := by
  sorry

end sunflower_seed_distribution_l2822_282283


namespace cubic_root_sum_l2822_282202

theorem cubic_root_sum (p q r : ℝ) : 
  (3 * p^3 - 5 * p^2 + 50 * p - 7 = 0) →
  (3 * q^3 - 5 * q^2 + 50 * q - 7 = 0) →
  (3 * r^3 - 5 * r^2 + 50 * r - 7 = 0) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 249/9 := by
sorry

end cubic_root_sum_l2822_282202


namespace initial_commission_rate_l2822_282228

theorem initial_commission_rate 
  (income_unchanged : ℝ → ℝ → ℝ → ℝ → Bool)
  (new_rate : ℝ)
  (business_slump : ℝ)
  (initial_rate : ℝ) :
  income_unchanged initial_rate new_rate business_slump initial_rate →
  new_rate = 5 →
  business_slump = 20.000000000000007 →
  initial_rate = 4 := by
sorry

end initial_commission_rate_l2822_282228


namespace integer_solutions_system_l2822_282200

theorem integer_solutions_system : 
  ∀ x y z : ℤ, 
    x^2 - y^2 - z^2 = 1 ∧ y + z - x = 3 →
    ((x = 9 ∧ y = 8 ∧ z = 4) ∨
     (x = -3 ∧ y = -2 ∧ z = 2) ∨
     (x = 9 ∧ y = 4 ∧ z = 8) ∨
     (x = -3 ∧ y = 2 ∧ z = -2)) :=
by
  sorry

end integer_solutions_system_l2822_282200


namespace no_intersection_absolute_value_graphs_l2822_282274

theorem no_intersection_absolute_value_graphs : 
  ∀ x : ℝ, ¬(|3 * x + 6| = -|4 * x - 1|) := by
  sorry

end no_intersection_absolute_value_graphs_l2822_282274


namespace min_value_of_expression_exists_min_value_l2822_282253

theorem min_value_of_expression (x : ℚ) : (2*x - 5)^2 + 18 ≥ 18 :=
sorry

theorem exists_min_value : ∃ x : ℚ, (2*x - 5)^2 + 18 = 18 :=
sorry

end min_value_of_expression_exists_min_value_l2822_282253


namespace midpoint_complex_numbers_l2822_282276

theorem midpoint_complex_numbers : 
  let A : ℂ := 1 / (1 + Complex.I)
  let B : ℂ := 1 / (1 - Complex.I)
  let C : ℂ := (A + B) / 2
  C = (1 : ℂ) / 2 := by sorry

end midpoint_complex_numbers_l2822_282276


namespace total_dogs_in_kennel_l2822_282237

-- Define the sets and their sizes
def T : ℕ := 45  -- Number of dogs with tags
def C : ℕ := 40  -- Number of dogs with collars
def B : ℕ := 6   -- Number of dogs with both tags and collars
def N : ℕ := 1   -- Number of dogs with neither tags nor collars

-- Theorem statement
theorem total_dogs_in_kennel : T + C - B + N = 80 := by
  sorry

end total_dogs_in_kennel_l2822_282237


namespace equal_probability_same_different_color_l2822_282252

theorem equal_probability_same_different_color (t : ℤ) :
  let n := t * (t + 1) / 2
  let k := t * (t - 1) / 2
  let total := n + k
  total ≥ 2 →
  (n * (n - 1) + k * (k - 1)) / (total * (total - 1)) = 
  (2 * n * k) / (total * (total - 1)) := by
sorry

end equal_probability_same_different_color_l2822_282252


namespace five_star_seven_l2822_282263

/-- The star operation defined as (a + b + 3)^2 -/
def star (a b : ℕ) : ℕ := (a + b + 3)^2

/-- Theorem stating that 5 ★ 7 = 225 -/
theorem five_star_seven : star 5 7 = 225 := by
  sorry

end five_star_seven_l2822_282263


namespace medium_box_tape_proof_l2822_282279

/-- The amount of tape (in feet) needed to seal a large box -/
def large_box_tape : ℝ := 4

/-- The amount of tape (in feet) needed to seal a small box -/
def small_box_tape : ℝ := 1

/-- The amount of tape (in feet) needed for the address label on any box -/
def label_tape : ℝ := 1

/-- The number of large boxes packed -/
def num_large_boxes : ℕ := 2

/-- The number of medium boxes packed -/
def num_medium_boxes : ℕ := 8

/-- The number of small boxes packed -/
def num_small_boxes : ℕ := 5

/-- The total amount of tape (in feet) used -/
def total_tape : ℝ := 44

/-- The amount of tape (in feet) needed to seal a medium box -/
def medium_box_tape : ℝ := 2

theorem medium_box_tape_proof :
  medium_box_tape * num_medium_boxes + 
  large_box_tape * num_large_boxes + 
  small_box_tape * num_small_boxes + 
  label_tape * (num_large_boxes + num_medium_boxes + num_small_boxes) = 
  total_tape := by sorry

end medium_box_tape_proof_l2822_282279


namespace space_diagonal_length_l2822_282206

/-- The length of the space diagonal in a rectangular prism with edge lengths 2, 3, and 4 is √29. -/
theorem space_diagonal_length (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 4) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 29 := by
  sorry


end space_diagonal_length_l2822_282206


namespace letter_lock_max_attempts_l2822_282227

/-- A letter lock with a given number of rings and letters per ring. -/
structure LetterLock :=
  (num_rings : ℕ)
  (letters_per_ring : ℕ)

/-- The maximum number of distinct unsuccessful attempts for a letter lock. -/
def max_unsuccessful_attempts (lock : LetterLock) : ℕ :=
  lock.letters_per_ring ^ lock.num_rings - 1

/-- Theorem: For a letter lock with 3 rings and 6 letters per ring,
    the maximum number of distinct unsuccessful attempts is 215. -/
theorem letter_lock_max_attempts :
  let lock := LetterLock.mk 3 6
  max_unsuccessful_attempts lock = 215 := by
  sorry

end letter_lock_max_attempts_l2822_282227


namespace solution_set_inequality_l2822_282213

theorem solution_set_inequality (x : ℝ) :
  (-x^2 + 3*x - 2 ≥ 0) ↔ (1 ≤ x ∧ x ≤ 2) := by sorry

end solution_set_inequality_l2822_282213


namespace south_cyclist_speed_l2822_282232

/-- The speed of a cyclist going south, given two cyclists start from the same place
    in opposite directions, one going north at 10 kmph, and they are 50 km apart after 1 hour. -/
def speed_of_south_cyclist : ℝ :=
  let speed_north : ℝ := 10
  let time : ℝ := 1
  let distance_apart : ℝ := 50
  distance_apart - speed_north * time

theorem south_cyclist_speed : speed_of_south_cyclist = 40 := by
  sorry

end south_cyclist_speed_l2822_282232


namespace city_population_ratio_l2822_282261

theorem city_population_ratio (pop_x pop_y pop_z : ℝ) (s : ℝ) 
  (h1 : pop_x = 6 * pop_y)
  (h2 : pop_y = s * pop_z)
  (h3 : pop_x / pop_z = 12)
  (h4 : pop_z > 0) : 
  pop_y / pop_z = 2 := by
sorry

end city_population_ratio_l2822_282261


namespace road_length_is_10km_l2822_282217

/-- Represents the road construction project -/
structure RoadProject where
  totalDays : ℕ
  initialWorkers : ℕ
  daysElapsed : ℕ
  completedLength : ℝ
  extraWorkers : ℕ

/-- Calculates the total length of the road given the project parameters -/
def calculateRoadLength (project : RoadProject) : ℝ :=
  sorry

/-- Theorem stating that the road length is 10 km given the specific project conditions -/
theorem road_length_is_10km (project : RoadProject) 
  (h1 : project.totalDays = 300)
  (h2 : project.initialWorkers = 30)
  (h3 : project.daysElapsed = 100)
  (h4 : project.completedLength = 2)
  (h5 : project.extraWorkers = 30) :
  calculateRoadLength project = 10 := by
  sorry

end road_length_is_10km_l2822_282217


namespace second_number_proof_l2822_282277

theorem second_number_proof (x : ℕ) : 
  (∃ k₁ k₂ : ℕ, 690 = 170 * k₁ + 10 ∧ x = 170 * k₂ + 25) ∧
  (∀ d : ℕ, d > 170 → ¬(∃ m₁ m₂ : ℕ, 690 = d * m₁ + 10 ∧ x = d * m₂ + 25)) →
  x = 875 := by
sorry

end second_number_proof_l2822_282277


namespace polygon_interior_angles_l2822_282256

/-- Sum of interior angles of a polygon with n sides -/
def sumInteriorAngles (n : ℕ) : ℝ := (n - 2) * 180

theorem polygon_interior_angles :
  (∀ n : ℕ, n ≥ 3 → sumInteriorAngles n = (n - 2) * 180) ∧
  sumInteriorAngles 6 = 720 ∧
  (∃ n : ℕ, n ≥ 3 ∧ (1/3) * sumInteriorAngles n = 300 ∧ n = 7) :=
sorry

end polygon_interior_angles_l2822_282256


namespace triangle_angle_B_l2822_282268

theorem triangle_angle_B (a b : ℝ) (A : ℝ) (h1 : a = 2) (h2 : b = 2 * Real.sqrt 3) (h3 : A = π / 6) :
  ∃ B : ℝ, (B = π / 3 ∨ B = 2 * π / 3) ∧
    a / Real.sin A = b / Real.sin B :=
by sorry

end triangle_angle_B_l2822_282268


namespace simplify_expression_l2822_282208

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - b^2 = 9*b^3 + 5*b^2 := by
  sorry

end simplify_expression_l2822_282208


namespace solve_luncheon_problem_l2822_282255

def luncheon_problem (no_shows : ℕ) (table_capacity : ℕ) (tables_needed : ℕ) : Prop :=
  let attendees := tables_needed * table_capacity
  let total_invited := no_shows + attendees
  total_invited = 18

theorem solve_luncheon_problem :
  luncheon_problem 12 3 2 :=
by
  sorry

end solve_luncheon_problem_l2822_282255


namespace company_profit_and_assignment_l2822_282242

/-- Represents the profit calculation for a company with two products. -/
def CompanyProfit (totalWorkers : ℕ) (profitA profitB : ℚ) (decreaseRate : ℚ) : Prop :=
  ∃ x : ℕ,
    x ≤ totalWorkers ∧
    let workersA := totalWorkers - x
    let outputA := 2 * workersA
    let outputB := x
    let profitPerUnitB := profitB - decreaseRate * x
    let totalProfitA := profitA * outputA
    let totalProfitB := profitPerUnitB * outputB
    totalProfitA = totalProfitB + 650 ∧
    totalProfitA + totalProfitB = 2650

/-- Represents the optimal worker assignment when introducing a third product. -/
def OptimalAssignment (totalWorkers : ℕ) (profitA profitB profitC : ℚ) (decreaseRate : ℚ) : Prop :=
  ∃ m : ℕ,
    m ≤ totalWorkers ∧
    let workersA := m
    let workersC := 2 * m
    let workersB := totalWorkers - workersA - workersC
    workersA + workersB + workersC = totalWorkers ∧
    let outputA := 2 * workersA
    let outputB := workersB
    let outputC := workersC
    outputA = outputC ∧
    let profitPerUnitB := profitB - decreaseRate * workersB
    let totalProfit := profitA * outputA + profitPerUnitB * outputB + profitC * outputC
    totalProfit = 2650 ∧
    m = 10

/-- Theorem stating the company's profit and optimal assignment. -/
theorem company_profit_and_assignment :
  CompanyProfit 65 15 120 2 ∧
  OptimalAssignment 65 15 120 30 2 :=
sorry

end company_profit_and_assignment_l2822_282242


namespace horner_method_v2_l2822_282258

def f (x : ℝ) : ℝ := 2*x^7 + x^6 + x^4 + x^2 + 1

def horner_v2 (x : ℝ) : ℝ := 
  let v0 := 2
  let v1 := 2*x + 1
  v1 * x

theorem horner_method_v2 : horner_v2 2 = 10 := by sorry

end horner_method_v2_l2822_282258


namespace f_derivative_at_zero_l2822_282223

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.cos x - Real.cos (3 * x)) / x else 0

theorem f_derivative_at_zero : 
  deriv f 0 = 4 := by sorry

end f_derivative_at_zero_l2822_282223


namespace solve_for_m_l2822_282211

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + 9*m

-- State the theorem
theorem solve_for_m : ∃ m : ℝ, f m 2 = 2 * g m 2 ∧ m = 0 := by sorry

end solve_for_m_l2822_282211


namespace max_principals_is_three_l2822_282220

/-- Represents the duration of a principal's term in years -/
def term_length : ℕ := 4

/-- Represents the period of interest in years -/
def period_length : ℕ := 9

/-- Calculates the maximum number of principals that can serve during the period -/
def max_principals : ℕ := 
  (period_length + term_length - 1) / term_length

/-- Theorem stating that the maximum number of principals is 3 -/
theorem max_principals_is_three : max_principals = 3 := by
  sorry

end max_principals_is_three_l2822_282220


namespace tangent_lines_range_l2822_282214

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Define the function g(x) = 2x^3 - 3x^2
def g (x : ℝ) : ℝ := 2*x^3 - 3*x^2

-- Theorem statement
theorem tangent_lines_range (t : ℝ) :
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ x : ℝ, f x - (f a + f' a * (x - a)) = 0 → x = a) ∧
    (∀ x : ℝ, f x - (f b + f' b * (x - b)) = 0 → x = b) ∧
    (∀ x : ℝ, f x - (f c + f' c * (x - c)) = 0 → x = c) ∧
    t = f a + f' a * (3 - a) ∧
    t = f b + f' b * (3 - b) ∧
    t = f c + f' c * (3 - c)) →
  -9 < t ∧ t < 8 :=
sorry

end tangent_lines_range_l2822_282214


namespace unique_solution_implies_a_zero_or_one_l2822_282264

/-- Given a real number a, define the set A as the solutions to ax^2 + 2x + 1 = 0 -/
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

/-- Theorem: If A(a) has exactly one element, then a = 0 or a = 1 -/
theorem unique_solution_implies_a_zero_or_one (a : ℝ) : (∃! x, x ∈ A a) → a = 0 ∨ a = 1 := by
  sorry

end unique_solution_implies_a_zero_or_one_l2822_282264


namespace max_value_sum_of_fractions_l2822_282240

theorem max_value_sum_of_fractions (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 / (1 + x^2) + y^2 / (1 + y^2) + z^2 / (1 + z^2) = 2) :
  x / (1 + x^2) + y / (1 + y^2) + z / (1 + z^2) ≤ Real.sqrt 2 :=
by sorry

end max_value_sum_of_fractions_l2822_282240


namespace negation_of_universal_proposition_l2822_282229

open Real

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > -1 → Real.log (x + 1) < x) ↔ (∃ x : ℝ, x > -1 ∧ Real.log (x + 1) ≥ x) :=
by sorry

end negation_of_universal_proposition_l2822_282229


namespace percentage_problem_l2822_282260

theorem percentage_problem (n : ℝ) : (0.1 * 0.3 * 0.5 * n = 90) → n = 6000 := by
  sorry

end percentage_problem_l2822_282260


namespace wang_lei_pastries_l2822_282245

/-- Represents the number of pastries in a large box -/
def large_box_pastries : ℕ := 32

/-- Represents the number of pastries in a small box -/
def small_box_pastries : ℕ := 15

/-- Represents the cost of a large box in yuan -/
def large_box_cost : ℚ := 85.6

/-- Represents the cost of a small box in yuan -/
def small_box_cost : ℚ := 46.8

/-- Represents the total amount spent by Wang Lei in yuan -/
def total_spent : ℚ := 654

/-- Represents the total number of boxes bought by Wang Lei -/
def total_boxes : ℕ := 9

/-- Theorem stating that Wang Lei got 237 pastries -/
theorem wang_lei_pastries : 
  ∃ (large_boxes small_boxes : ℕ), 
    large_boxes + small_boxes = total_boxes ∧
    large_box_cost * large_boxes + small_box_cost * small_boxes = total_spent ∧
    large_box_pastries * large_boxes + small_box_pastries * small_boxes = 237 :=
by sorry

end wang_lei_pastries_l2822_282245


namespace exactly_two_valid_sets_l2822_282278

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)  -- The first integer in the set
  (length : ℕ) -- The number of integers in the set

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a valid set according to our conditions -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 3 ∧ sum_consecutive s = 18

/-- The main theorem to prove -/
theorem exactly_two_valid_sets :
  ∃! (sets : Finset ConsecutiveSet), sets.card = 2 ∧ ∀ s ∈ sets, is_valid_set s :=
sorry

end exactly_two_valid_sets_l2822_282278


namespace coin_flip_probability_l2822_282219

theorem coin_flip_probability :
  let n : ℕ := 5  -- total number of coins
  let k : ℕ := 3  -- number of specific coins we want to be heads
  let total_outcomes : ℕ := 2^n
  let favorable_outcomes : ℕ := 2^(n - k)
  favorable_outcomes / total_outcomes = (1 : ℚ) / 8 :=
by sorry

end coin_flip_probability_l2822_282219


namespace dark_tile_fraction_l2822_282226

theorem dark_tile_fraction (block_size : ℕ) (dark_tiles : ℕ) : 
  block_size = 8 → 
  dark_tiles = 18 → 
  (dark_tiles : ℚ) / (block_size * block_size : ℚ) = 9 / 32 :=
by sorry

end dark_tile_fraction_l2822_282226


namespace three_km_to_meters_four_kg_to_grams_l2822_282254

-- Define the conversion factors
def meters_per_kilometer : ℝ := 1000
def grams_per_kilogram : ℝ := 1000

-- Theorem for kilometer to meter conversion
theorem three_km_to_meters :
  3 * meters_per_kilometer = 3000 := by sorry

-- Theorem for kilogram to gram conversion
theorem four_kg_to_grams :
  4 * grams_per_kilogram = 4000 := by sorry

end three_km_to_meters_four_kg_to_grams_l2822_282254


namespace point_on_x_axis_l2822_282247

/-- If point P with coordinates (4-a, 3a+9) lies on the x-axis, then its coordinates are (7, 0) -/
theorem point_on_x_axis (a : ℝ) :
  let P : ℝ × ℝ := (4 - a, 3 * a + 9)
  (P.2 = 0) → P = (7, 0) := by
  sorry

end point_on_x_axis_l2822_282247
