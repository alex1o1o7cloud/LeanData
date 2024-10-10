import Mathlib

namespace smallest_year_after_2010_with_digit_sum_16_l1364_136456

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Predicate to check if a year is after 2010 -/
def is_after_2010 (year : ℕ) : Prop :=
  year > 2010

/-- Theorem stating that 2059 is the smallest year after 2010 with digit sum 16 -/
theorem smallest_year_after_2010_with_digit_sum_16 :
  (∀ year : ℕ, is_after_2010 year → sum_of_digits year = 16 → year ≥ 2059) ∧
  (is_after_2010 2059 ∧ sum_of_digits 2059 = 16) :=
sorry

end smallest_year_after_2010_with_digit_sum_16_l1364_136456


namespace guessing_game_l1364_136494

theorem guessing_game (G C : ℕ) (h1 : G = 33) (h2 : 3 * G = 2 * C - 3) : C = 51 := by
  sorry

end guessing_game_l1364_136494


namespace min_value_of_function_min_value_achievable_l1364_136464

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (x^2 + 3*x + 1) / x ≥ 5 :=
by sorry

theorem min_value_achievable :
  ∃ x : ℝ, x > 0 ∧ (x^2 + 3*x + 1) / x = 5 :=
by sorry

end min_value_of_function_min_value_achievable_l1364_136464


namespace rebus_puzzle_solution_l1364_136482

theorem rebus_puzzle_solution :
  ∃! (A B C : ℕ),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    100 * A + 10 * B + A + 100 * A + 10 * B + C = 100 * A + 10 * C + C ∧
    100 * A + 10 * C + C = 1416 ∧
    A = 4 ∧ B = 7 ∧ C = 6 :=
by sorry

end rebus_puzzle_solution_l1364_136482


namespace power_division_l1364_136499

theorem power_division (m : ℝ) : m^4 / m^2 = m^2 := by
  sorry

end power_division_l1364_136499


namespace square_difference_l1364_136493

theorem square_difference (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := by
  sorry

end square_difference_l1364_136493


namespace rectangle_problem_l1364_136472

/-- Given a rectangle with length 4x + 1 and width x + 7, where the area is equal to twice the perimeter,
    the positive value of x is (-9 + √481) / 8. -/
theorem rectangle_problem (x : ℝ) : 
  (4*x + 1) * (x + 7) = 2 * (2*(4*x + 1) + 2*(x + 7)) → 
  x > 0 → 
  x = (-9 + Real.sqrt 481) / 8 := by
sorry

end rectangle_problem_l1364_136472


namespace sequence_eventually_periodic_l1364_136455

def sequence_next (a : ℕ) (un : ℕ) : ℕ :=
  if un % 2 = 0 then un / 2 else a + un

def is_periodic (s : ℕ → ℕ) (k p : ℕ) : Prop :=
  ∀ n, n ≥ k → s (n + p) = s n

theorem sequence_eventually_periodic (a : ℕ) (h_a : Odd a) (u : ℕ → ℕ) 
  (h_u : ∀ n, u (n + 1) = sequence_next a (u n)) :
  ∃ k p, p > 0 ∧ is_periodic u k p :=
sorry

end sequence_eventually_periodic_l1364_136455


namespace total_cookies_l1364_136429

def num_bags : ℕ := 37
def cookies_per_bag : ℕ := 19

theorem total_cookies : num_bags * cookies_per_bag = 703 := by
  sorry

end total_cookies_l1364_136429


namespace rectangular_solid_volume_l1364_136490

theorem rectangular_solid_volume (a b c : ℝ) 
  (h1 : a * b = 18)
  (h2 : b * c = 50)
  (h3 : a * c = 45) :
  a * b * c = 150 * Real.sqrt 3 := by
sorry

end rectangular_solid_volume_l1364_136490


namespace brent_baby_ruths_l1364_136486

/-- The number of Baby Ruths Brent received -/
def baby_ruths : ℕ := sorry

/-- The number of Kit-Kat bars Brent received -/
def kit_kat : ℕ := 5

/-- The number of Hershey kisses Brent received -/
def hershey_kisses : ℕ := 3 * kit_kat

/-- The number of Nerds boxes Brent received -/
def nerds : ℕ := 8

/-- The number of lollipops Brent received -/
def lollipops : ℕ := 11

/-- The number of Reese Peanut butter cups Brent received -/
def reese_cups : ℕ := baby_ruths / 2

/-- The number of lollipops Brent gave to his sister -/
def lollipops_given : ℕ := 5

/-- The total number of candies Brent had left after giving lollipops to his sister -/
def total_left : ℕ := 49

theorem brent_baby_ruths :
  kit_kat + hershey_kisses + nerds + (lollipops - lollipops_given) + baby_ruths + reese_cups = total_left ∧
  baby_ruths = 10 := by sorry

end brent_baby_ruths_l1364_136486


namespace no_solution_iff_a_equals_two_l1364_136414

theorem no_solution_iff_a_equals_two (a : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (a * x) / (x - 1) + 3 / (1 - x) ≠ 2) ↔ a = 2 := by
sorry

end no_solution_iff_a_equals_two_l1364_136414


namespace train_delay_l1364_136453

/-- Proves that a train moving at 6/7 of its usual speed will be 30 minutes late on a journey that usually takes 3 hours. -/
theorem train_delay (usual_speed : ℝ) (usual_time : ℝ) (h1 : usual_time = 3) :
  let current_speed := (6/7) * usual_speed
  let current_time := usual_speed * usual_time / current_speed
  (current_time - usual_time) * 60 = 30 := by
  sorry

end train_delay_l1364_136453


namespace cinnamon_tradition_duration_l1364_136476

/-- Represents the cinnamon ball tradition setup -/
structure CinnamonTradition where
  totalSocks : Nat
  extraSocks : Nat
  regularBalls : Nat
  extraBalls : Nat
  totalBalls : Nat

/-- Calculates the maximum number of full days the tradition can continue -/
def maxDays (ct : CinnamonTradition) : Nat :=
  ct.totalBalls / (ct.regularBalls * (ct.totalSocks - ct.extraSocks) + ct.extraBalls * ct.extraSocks)

/-- Theorem stating that for the given conditions, the tradition lasts 3 days -/
theorem cinnamon_tradition_duration :
  ∀ (ct : CinnamonTradition),
  ct.totalSocks = 9 →
  ct.extraSocks = 3 →
  ct.regularBalls = 2 →
  ct.extraBalls = 3 →
  ct.totalBalls = 75 →
  maxDays ct = 3 := by
  sorry

#eval maxDays { totalSocks := 9, extraSocks := 3, regularBalls := 2, extraBalls := 3, totalBalls := 75 }

end cinnamon_tradition_duration_l1364_136476


namespace binomial_expansion_with_arithmetic_sequence_coefficients_l1364_136483

/-- 
Given a binomial expansion (a+b)^n where the coefficients of the first three terms 
form an arithmetic sequence, this theorem proves that n = 8 and identifies 
the rational terms in the expansion when a = x and b = 1/2.
-/
theorem binomial_expansion_with_arithmetic_sequence_coefficients :
  ∀ n : ℕ,
  (∃ d : ℚ, (n.choose 1 : ℚ) = (n.choose 0 : ℚ) + d ∧ (n.choose 2 : ℚ) = (n.choose 1 : ℚ) + d) →
  (n = 8 ∧ 
   ∀ r : ℕ, r ≤ n → 
   (r = 0 ∨ r = 4 ∨ r = 8) ↔ ∃ q : ℚ, (n.choose r : ℚ) * (1 / 2 : ℚ)^r = q) :=
by sorry


end binomial_expansion_with_arithmetic_sequence_coefficients_l1364_136483


namespace aquafaba_to_egg_white_ratio_l1364_136492

theorem aquafaba_to_egg_white_ratio : 
  let num_cakes : ℕ := 2
  let egg_whites_per_cake : ℕ := 8
  let total_aquafaba : ℕ := 32
  let total_egg_whites : ℕ := num_cakes * egg_whites_per_cake
  (total_aquafaba : ℚ) / (total_egg_whites : ℚ) = 2 := by
  sorry

end aquafaba_to_egg_white_ratio_l1364_136492


namespace equation_solution_l1364_136496

theorem equation_solution (x : ℝ) : 
  (8 * x^2 + 120 * x + 7) / (3 * x + 10) = 4 * x + 2 ↔ 
  -4 * x^2 + 74 * x - 13 = 0 :=
by sorry

end equation_solution_l1364_136496


namespace arctan_sum_equation_n_unique_l1364_136434

/-- The positive integer n satisfying the equation arctan(1/2) + arctan(1/3) + arctan(1/7) + arctan(1/n) = π/4 -/
def n : ℕ := 7

/-- The equation that n satisfies -/
theorem arctan_sum_equation : 
  Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/n) = π/4 := by
  sorry

/-- Proof that n is the unique positive integer satisfying the equation -/
theorem n_unique : 
  ∀ m : ℕ, m > 0 → 
  (Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/m) = π/4) → 
  m = n := by
  sorry

end arctan_sum_equation_n_unique_l1364_136434


namespace parity_of_f_l1364_136421

/-- A function that is not always zero -/
def NonZeroFunction (f : ℝ → ℝ) : Prop :=
  ∃ x, f x ≠ 0

/-- Definition of an odd function -/
def OddFunction (F : ℝ → ℝ) : Prop :=
  ∀ x, F (-x) = -F x

/-- Definition of an even function -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The main theorem -/
theorem parity_of_f (f : ℝ → ℝ) (h_nonzero : NonZeroFunction f) :
    let F := fun x => if x ≠ 0 then (x^3 - 2*x) * f x else 0
    OddFunction F → EvenFunction f := by
  sorry

end parity_of_f_l1364_136421


namespace curve_E_and_min_distance_l1364_136432

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 5 = 0

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 91 = 0

/-- Definition of curve E as the locus of centers of moving circles -/
def E (x y : ℝ) : Prop := ∃ (r : ℝ), 
  (∀ (x₁ y₁ : ℝ), C₁ x₁ y₁ → (x - x₁)^2 + (y - y₁)^2 = (r + 2)^2) ∧
  (∀ (x₂ y₂ : ℝ), C₂ x₂ y₂ → (x - x₂)^2 + (y - y₂)^2 = (10 - r)^2)

/-- The right focus of curve E -/
def F : ℝ × ℝ := (3, 0)

/-- Theorem stating the equation of curve E and the minimum value of |PO|²+|PF|² -/
theorem curve_E_and_min_distance : 
  (∀ x y : ℝ, E x y ↔ x^2/36 + y^2/27 = 1) ∧
  (∃ min : ℝ, min = 45 ∧ 
    ∀ x y : ℝ, E x y → x^2 + y^2 + (x - F.1)^2 + (y - F.2)^2 ≥ min) :=
sorry

end curve_E_and_min_distance_l1364_136432


namespace hexagon_division_l1364_136409

/-- A regular hexagon with all sides and diagonals drawn -/
structure RegularHexagonWithDiagonals where
  /-- The number of vertices in a regular hexagon -/
  num_vertices : Nat
  /-- The number of sides in a regular hexagon -/
  num_sides : Nat
  /-- The number of diagonals in a regular hexagon -/
  num_diagonals : Nat
  /-- Assertion that the number of vertices is 6 -/
  vertex_count : num_vertices = 6
  /-- Assertion that the number of sides is equal to the number of vertices -/
  side_count : num_sides = num_vertices
  /-- Formula for the number of diagonals in a hexagon -/
  diagonal_count : num_diagonals = (num_vertices * (num_vertices - 3)) / 2

/-- The number of regions into which a regular hexagon is divided when all its sides and diagonals are drawn -/
def num_regions (h : RegularHexagonWithDiagonals) : Nat := 24

/-- Theorem stating that drawing all sides and diagonals of a regular hexagon divides it into 24 regions -/
theorem hexagon_division (h : RegularHexagonWithDiagonals) : num_regions h = 24 := by
  sorry

end hexagon_division_l1364_136409


namespace set_operation_result_l1364_136428

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 2, 3}

-- Define set B
def B : Set Nat := {3, 4, 5}

-- Theorem to prove
theorem set_operation_result :
  ((U \ A) ∩ B) = {4, 5} := by
  sorry

end set_operation_result_l1364_136428


namespace sunset_colors_l1364_136468

/-- The number of colors the sky turns during a sunset -/
def sky_colors (sunset_duration : ℕ) (color_change_interval : ℕ) (minutes_per_hour : ℕ) : ℕ :=
  (sunset_duration * minutes_per_hour) / color_change_interval

/-- Theorem: During a 2-hour sunset, with the sky changing color every 10 minutes,
    and each hour being 60 minutes long, the sky turns 12 different colors. -/
theorem sunset_colors :
  sky_colors 2 10 60 = 12 := by
  sorry

#eval sky_colors 2 10 60

end sunset_colors_l1364_136468


namespace unique_three_digit_number_l1364_136458

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds_bound : hundreds < 10
  h_tens_bound : tens < 10
  h_ones_bound : ones < 10

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The reverse of a three-digit number -/
def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : Nat :=
  100 * n.ones + 10 * n.tens + n.hundreds

theorem unique_three_digit_number :
  ∃! n : ThreeDigitNumber,
    (n.hundreds + n.tens + n.ones = 10) ∧
    (n.tens = n.hundreds + n.ones) ∧
    (n.reverse = n.value + 99) ∧
    (n.value = 253) := by
  sorry

end unique_three_digit_number_l1364_136458


namespace min_value_sum_product_l1364_136465

theorem min_value_sum_product (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b + c) + 1 / (a + b + d) + 1 / (a + c + d) + 1 / (b + c + d)) ≥ 16 / 3 := by
  sorry

end min_value_sum_product_l1364_136465


namespace product_expansion_l1364_136441

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end product_expansion_l1364_136441


namespace two_positive_real_roots_condition_no_real_roots_necessary_condition_l1364_136401

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : Prop := x^2 + (m - 3) * x + m = 0

-- Define the condition for two positive real roots
def has_two_positive_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
  quadratic_equation m x₁ ∧ quadratic_equation m x₂

-- Define the condition for no real roots
def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, ¬(quadratic_equation m x)

-- Theorem for two positive real roots
theorem two_positive_real_roots_condition :
  ∀ m : ℝ, has_two_positive_real_roots m ↔ (0 < m ∧ m ≤ 1) :=
sorry

-- Theorem for necessary condition of no real roots
theorem no_real_roots_necessary_condition :
  ∀ m : ℝ, has_no_real_roots m → m > 1 :=
sorry

end two_positive_real_roots_condition_no_real_roots_necessary_condition_l1364_136401


namespace min_value_expression_lower_bound_achievable_l1364_136471

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (((x^2 + y^2 + z^2) * (4*x^2 + 2*y^2 + 3*z^2)).sqrt) / (x*y*z) ≥ 2 + Real.sqrt 2 + Real.sqrt 3 :=
sorry

theorem lower_bound_achievable :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (((x^2 + y^2 + z^2) * (4*x^2 + 2*y^2 + 3*z^2)).sqrt) / (x*y*z) = 2 + Real.sqrt 2 + Real.sqrt 3 :=
sorry

end min_value_expression_lower_bound_achievable_l1364_136471


namespace compound_molecular_weight_l1364_136435

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (al_count : ℕ) (o_count : ℕ) (h_count : ℕ) 
  (al_weight : ℝ) (o_weight : ℝ) (h_weight : ℝ) : ℝ :=
  al_count * al_weight + o_count * o_weight + h_count * h_weight

/-- The molecular weight of the compound AlO₃H₃ is 78.01 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight 1 3 3 26.98 16.00 1.01 = 78.01 := by
  sorry

end compound_molecular_weight_l1364_136435


namespace complex_multiplication_l1364_136412

theorem complex_multiplication (i : ℂ) :
  i^2 = -1 →
  (3 - 4*i) * (-6 + 2*i) = -10 + 30*i := by sorry

end complex_multiplication_l1364_136412


namespace smallest_prime_digit_sum_23_l1364_136489

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem: 1997 is the smallest prime whose digits sum to 23 -/
theorem smallest_prime_digit_sum_23 :
  (is_prime 1997) ∧ 
  (digit_sum 1997 = 23) ∧ 
  (∀ n : ℕ, n < 1997 → (is_prime n ∧ digit_sum n = 23) → False) :=
sorry

end smallest_prime_digit_sum_23_l1364_136489


namespace color_infinite_lines_parallelogram_property_coloring_theorem_l1364_136439

-- Define the color type
inductive Color where
  | White : Color
  | Red : Color
  | Black : Color

-- Define the coloring function
def f : ℤ × ℤ → Color :=
  sorry

-- Condition 1: Each color appears on infinitely many horizontal lines
theorem color_infinite_lines :
  ∀ c : Color, ∃ (s : Set ℤ), Set.Infinite s ∧
    ∀ y ∈ s, ∃ (t : Set ℤ), Set.Infinite t ∧
      ∀ x ∈ t, f (x, y) = c :=
  sorry

-- Condition 2: Parallelogram property
theorem parallelogram_property :
  ∀ a b c : ℤ × ℤ,
    f a = Color.White → f b = Color.Red → f c = Color.Black →
    ∃ d : ℤ × ℤ, f d = Color.Red ∧ a + c = b + d :=
  sorry

-- Main theorem combining both conditions
theorem coloring_theorem :
  ∃ (f : ℤ × ℤ → Color),
    (∀ c : Color, ∃ (s : Set ℤ), Set.Infinite s ∧
      ∀ y ∈ s, ∃ (t : Set ℤ), Set.Infinite t ∧
        ∀ x ∈ t, f (x, y) = c) ∧
    (∀ a b c : ℤ × ℤ,
      f a = Color.White → f b = Color.Red → f c = Color.Black →
      ∃ d : ℤ × ℤ, f d = Color.Red ∧ a + c = b + d) :=
  sorry

end color_infinite_lines_parallelogram_property_coloring_theorem_l1364_136439


namespace inequality_solution_sets_l1364_136475

theorem inequality_solution_sets (a : ℝ) : 
  (∀ x : ℝ, 3 * x - 5 < a ↔ 2 * x < 4) → a = 1 := by
  sorry

end inequality_solution_sets_l1364_136475


namespace orthocenter_of_specific_triangle_l1364_136430

/-- The orthocenter of a triangle ABC in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (5/3, 29/3, 8/3) -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 1)
  let B : ℝ × ℝ × ℝ := (4, -1, 5)
  let C : ℝ × ℝ × ℝ := (1, 5, 2)
  orthocenter A B C = (5/3, 29/3, 8/3) := by sorry

end orthocenter_of_specific_triangle_l1364_136430


namespace parabola_directrix_l1364_136420

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = -17/4

/-- Theorem: The directrix of the given parabola is y = -17/4 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola x y → ∃ (d : ℝ), directrix d ∧ 
  (∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y → 
    (p.1 - 4)^2 + (p.2 - d)^2 = (p.2 - (d + 4))^2) :=
sorry

end parabola_directrix_l1364_136420


namespace vanessa_picked_17_carrots_l1364_136462

/-- The number of carrots Vanessa picked -/
def vanessas_carrots (good_carrots bad_carrots moms_carrots : ℕ) : ℕ :=
  good_carrots + bad_carrots - moms_carrots

/-- Proof that Vanessa picked 17 carrots -/
theorem vanessa_picked_17_carrots :
  vanessas_carrots 24 7 14 = 17 := by
  sorry

end vanessa_picked_17_carrots_l1364_136462


namespace function_range_l1364_136470

theorem function_range (x : ℝ) :
  (∀ a ∈ Set.Icc (-1 : ℝ) 1, x^2 + (a - 4)*x + 4 - 2*a > 0) →
  x < 1 ∨ x > 3 := by
  sorry

end function_range_l1364_136470


namespace rectangle_to_square_l1364_136406

theorem rectangle_to_square (k : ℕ) (h1 : k > 5) : 
  (∃ (n : ℕ), k * (k - 5) = n^2) → k * (k - 5) = 6^2 :=
by sorry

end rectangle_to_square_l1364_136406


namespace inequality_holds_for_all_real_x_l1364_136447

theorem inequality_holds_for_all_real_x : ∀ x : ℝ, 2^(Real.sin x)^2 + 2^(Real.cos x)^2 ≥ 2 * Real.sqrt 2 := by
  sorry

end inequality_holds_for_all_real_x_l1364_136447


namespace log_sum_property_l1364_136438

theorem log_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
  sorry

end log_sum_property_l1364_136438


namespace parabola_midpoint_distance_l1364_136418

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = x^2 + 3*x + 2

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

-- Define the square of the distance between two points
def square_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2

-- Theorem statement
theorem parabola_midpoint_distance 
  (C D : PointOnParabola) 
  (h : is_midpoint C.x C.y D.x D.y) : 
  square_distance C.x C.y D.x D.y = 16 := by
  sorry

end parabola_midpoint_distance_l1364_136418


namespace turtleneck_profit_l1364_136461

/-- Represents the pricing strategy and profit calculation for turtleneck sweaters -/
theorem turtleneck_profit (C : ℝ) (C_pos : C > 0) : 
  let initial_markup : ℝ := 0.20
  let new_year_markup : ℝ := 0.25
  let february_discount : ℝ := 0.09
  let SP1 : ℝ := C * (1 + initial_markup)
  let SP2 : ℝ := SP1 * (1 + new_year_markup)
  let SPF : ℝ := SP2 * (1 - february_discount)
  let profit : ℝ := SPF - C
  profit / C = 0.365 := by sorry

end turtleneck_profit_l1364_136461


namespace tank_capacity_is_120_gallons_l1364_136404

/-- Represents the capacity of a water tank in gallons -/
def tank_capacity : ℝ := 120

/-- Represents the difference in gallons between 70% and 40% full -/
def difference : ℝ := 36

/-- Theorem stating that the tank capacity is 120 gallons -/
theorem tank_capacity_is_120_gallons : 
  (0.7 * tank_capacity - 0.4 * tank_capacity = difference) → 
  tank_capacity = 120 := by
  sorry

end tank_capacity_is_120_gallons_l1364_136404


namespace f_g_f_3_equals_630_l1364_136448

def f (x : ℝ) : ℝ := 5 * x + 5

def g (x : ℝ) : ℝ := 6 * x + 5

theorem f_g_f_3_equals_630 : f (g (f 3)) = 630 := by
  sorry

end f_g_f_3_equals_630_l1364_136448


namespace students_allowance_l1364_136436

theorem students_allowance (allowance : ℚ) : 
  (2 / 3 : ℚ) * (2 / 5 : ℚ) * allowance = 6 / 10 → 
  allowance = 9 / 4 := by
sorry

end students_allowance_l1364_136436


namespace milburg_grown_ups_l1364_136442

/-- The population of Milburg -/
def total_population : ℕ := 8243

/-- The number of children in Milburg -/
def children : ℕ := 2987

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := total_population - children

/-- Theorem stating that the number of grown-ups in Milburg is 5256 -/
theorem milburg_grown_ups : grown_ups = 5256 := by
  sorry

end milburg_grown_ups_l1364_136442


namespace race_distance_l1364_136424

/-- Represents the race scenario with given conditions -/
structure RaceScenario where
  distance : ℝ
  timeA : ℝ
  startAdvantage1 : ℝ
  timeDifference : ℝ
  startAdvantage2 : ℝ

/-- Defines the conditions of the race -/
def raceConditions : RaceScenario → Prop
  | ⟨d, t, s1, dt, s2⟩ => 
    t = 77.5 ∧ 
    s1 = 25 ∧ 
    dt = 10 ∧ 
    s2 = 45 ∧ 
    d / t = (d - s1) / (t + dt) ∧ 
    d / t = (d - s2) / t

/-- Theorem stating that the race distance is 218.75 meters -/
theorem race_distance (scenario : RaceScenario) 
  (h : raceConditions scenario) : scenario.distance = 218.75 := by
  sorry

#check race_distance

end race_distance_l1364_136424


namespace stratified_sampling_l1364_136425

/-- Stratified sampling problem -/
theorem stratified_sampling 
  (total_employees : ℕ) 
  (middle_managers : ℕ) 
  (senior_managers : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 150) 
  (h2 : middle_managers = 30) 
  (h3 : senior_managers = 10) 
  (h4 : sample_size = 30) :
  (sample_size * middle_managers / total_employees = 6) ∧ 
  (sample_size * senior_managers / total_employees = 2) :=
by sorry

end stratified_sampling_l1364_136425


namespace quadratic_no_real_roots_l1364_136450

/-- A sufficient but not necessary condition for a quadratic function to have no real roots -/
theorem quadratic_no_real_roots (a b c : ℝ) (ha : a ≠ 0) :
  b^2 - 4*a*c < -1 → ∀ x, a*x^2 + b*x + c ≠ 0 := by sorry

end quadratic_no_real_roots_l1364_136450


namespace triangle_area_l1364_136437

/-- The area of a triangle with base 12 and height 5 is 30 -/
theorem triangle_area : 
  let base : ℝ := 12
  let height : ℝ := 5
  (1/2 : ℝ) * base * height = 30 := by sorry

end triangle_area_l1364_136437


namespace min_value_expression_l1364_136451

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^2 + 8*a*b + 24*b^2 + 16*b*c + 6*c^2 ≥ 18 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 1 ∧
    a₀^2 + 8*a₀*b₀ + 24*b₀^2 + 16*b₀*c₀ + 6*c₀^2 = 18 := by
  sorry

end min_value_expression_l1364_136451


namespace units_digit_of_27_times_36_l1364_136433

theorem units_digit_of_27_times_36 : (27 * 36) % 10 = 2 := by
  sorry

end units_digit_of_27_times_36_l1364_136433


namespace difference_between_sum_and_average_l1364_136479

def numbers : List ℕ := [44, 16, 2, 77, 241]

theorem difference_between_sum_and_average : 
  (numbers.sum : ℚ) - (numbers.sum : ℚ) / numbers.length = 304 := by
  sorry

end difference_between_sum_and_average_l1364_136479


namespace intersection_of_A_and_B_l1364_136488

def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by sorry

end intersection_of_A_and_B_l1364_136488


namespace largest_geometric_sequence_number_l1364_136491

/-- Checks if a three-digit number's digits form a geometric sequence -/
def is_geometric_sequence (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : ℕ) (r : ℚ),
    n = 100 * a + 10 * b + c ∧
    a = 8 ∧
    b = Int.floor (8 * r) ∧
    c = Int.floor (8 * r^2) ∧
    r > 0 ∧ r < 1

/-- Checks if a three-digit number has distinct digits -/
def has_distinct_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The main theorem stating that 842 is the largest three-digit number
    satisfying the given conditions -/
theorem largest_geometric_sequence_number :
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 →
    is_geometric_sequence n →
    has_distinct_digits n →
    n ≤ 842 :=
by sorry

end largest_geometric_sequence_number_l1364_136491


namespace five_people_seven_chairs_l1364_136495

/-- The number of ways to arrange n people in r chairs -/
def arrangements (n r : ℕ) : ℕ := sorry

/-- The number of ways to arrange n people in r chairs,
    with one person restricted to m specific chairs -/
def restricted_arrangements (n r m : ℕ) : ℕ := sorry

/-- Theorem: Five people can be arranged in a row of seven chairs in 2160 ways,
    given that the oldest must sit in one of the three chairs at the end of the row -/
theorem five_people_seven_chairs : restricted_arrangements 5 7 3 = 2160 := by sorry

end five_people_seven_chairs_l1364_136495


namespace data_mode_is_60_l1364_136474

def data : List Nat := [65, 60, 75, 60, 80]

def mode (l : List Nat) : Nat :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem data_mode_is_60 : mode data = 60 := by
  sorry

end data_mode_is_60_l1364_136474


namespace walter_age_theorem_l1364_136407

/-- Walter's age at the end of 1998 -/
def walter_age_1998 : ℕ := 34

/-- Walter's grandmother's age at the end of 1998 -/
def grandmother_age_1998 : ℕ := 3 * walter_age_1998

/-- The sum of Walter's and his grandmother's birth years -/
def birth_years_sum : ℕ := 3860

/-- Walter's age at the end of 2003 -/
def walter_age_2003 : ℕ := walter_age_1998 + 5

theorem walter_age_theorem :
  (1998 - walter_age_1998) + (1998 - grandmother_age_1998) = birth_years_sum ∧
  walter_age_2003 = 39 := by
  sorry

end walter_age_theorem_l1364_136407


namespace probability_of_selection_l1364_136454

/-- The number of shirts in the drawer -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the drawer -/
def num_shorts : ℕ := 7

/-- The number of pairs of socks in the drawer -/
def num_socks : ℕ := 8

/-- The total number of articles of clothing -/
def total_items : ℕ := num_shirts + num_shorts + num_socks

/-- The number of items to be selected -/
def items_selected : ℕ := 5

/-- The probability of selecting two shirts, two pairs of shorts, and one pair of socks -/
theorem probability_of_selection : 
  (Nat.choose num_shirts 2 * Nat.choose num_shorts 2 * Nat.choose num_socks 1) / 
  Nat.choose total_items items_selected = 280 / 2261 := by
  sorry

end probability_of_selection_l1364_136454


namespace initial_fee_value_l1364_136416

/-- The initial fee of the first car rental plan -/
def initial_fee : ℝ := sorry

/-- The cost per mile for the first car rental plan -/
def cost_per_mile_plan1 : ℝ := 0.40

/-- The cost per mile for the second car rental plan -/
def cost_per_mile_plan2 : ℝ := 0.60

/-- The number of miles driven for which both plans cost the same -/
def miles_driven : ℝ := 325

theorem initial_fee_value :
  initial_fee = 65 :=
by
  have h1 : initial_fee + cost_per_mile_plan1 * miles_driven = cost_per_mile_plan2 * miles_driven :=
    sorry
  sorry

end initial_fee_value_l1364_136416


namespace inequality_system_solution_l1364_136403

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2*x ∧ x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by sorry

end inequality_system_solution_l1364_136403


namespace difference_in_cost_l1364_136446

def joy_pencils : ℕ := 30
def colleen_pencils : ℕ := 50
def pencil_cost : ℕ := 4

theorem difference_in_cost : (colleen_pencils - joy_pencils) * pencil_cost = 80 := by
  sorry

end difference_in_cost_l1364_136446


namespace basketball_match_loss_percentage_l1364_136444

theorem basketball_match_loss_percentage 
  (won lost : ℕ) 
  (h1 : won > 0 ∧ lost > 0) 
  (h2 : won / lost = 7 / 3) : 
  (lost : ℚ) / ((won : ℚ) + lost) * 100 = 30 := by
sorry

end basketball_match_loss_percentage_l1364_136444


namespace sum_of_digits_n_plus_5_l1364_136473

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_n_plus_5 (n : ℕ) (h1 : S n = 365) (h2 : n % 8 = S n % 8) :
  S (n + 5) = 370 := by
  sorry

end sum_of_digits_n_plus_5_l1364_136473


namespace circle_area_theorem_l1364_136469

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (4, 10)
def B : ℝ × ℝ := (10, 8)

-- State that A and B lie on circle ω
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω

-- Define the tangent lines at A and B
def tangent_A : Set (ℝ × ℝ) := sorry
def tangent_B : Set (ℝ × ℝ) := sorry

-- Define the intersection point of tangent lines
def intersection : ℝ × ℝ := sorry

-- State that the intersection point is on the x-axis
axiom intersection_on_x_axis : intersection.2 = 0

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem circle_area_theorem : circle_area ω = 100 * π / 9 := by sorry

end circle_area_theorem_l1364_136469


namespace probability_red_from_box2_is_11_27_l1364_136405

/-- Represents a box containing balls of two colors -/
structure Box where
  white : ℕ
  red : ℕ

/-- The probability of drawing a red ball from box 2 after the described process -/
def probability_red_from_box2 (box1 box2 : Box) : ℚ :=
  let total_balls1 := box1.white + box1.red
  let total_balls2 := box2.white + box2.red + 1
  let prob_white_from_box1 := box1.white / total_balls1
  let prob_red_from_box1 := box1.red / total_balls1
  let prob_red_if_white_moved := prob_white_from_box1 * (box2.red / total_balls2)
  let prob_red_if_red_moved := prob_red_from_box1 * ((box2.red + 1) / total_balls2)
  prob_red_if_white_moved + prob_red_if_red_moved

theorem probability_red_from_box2_is_11_27 :
  let box1 : Box := { white := 2, red := 4 }
  let box2 : Box := { white := 5, red := 3 }
  probability_red_from_box2 box1 box2 = 11/27 := by
  sorry

end probability_red_from_box2_is_11_27_l1364_136405


namespace tim_dozens_of_golf_balls_l1364_136497

def total_golf_balls : ℕ := 156
def balls_per_dozen : ℕ := 12

theorem tim_dozens_of_golf_balls : 
  total_golf_balls / balls_per_dozen = 13 := by sorry

end tim_dozens_of_golf_balls_l1364_136497


namespace average_score_is_correct_total_students_is_correct_l1364_136413

/-- Calculates the average score given a list of (score, number of students) pairs -/
def averageScore (scores : List (ℚ × ℕ)) : ℚ :=
  let totalScore := scores.foldl (fun acc (score, count) => acc + score * count) 0
  let totalStudents := scores.foldl (fun acc (_, count) => acc + count) 0
  totalScore / totalStudents

/-- The given score distribution -/
def scoreDistribution : List (ℚ × ℕ) :=
  [(100, 10), (95, 20), (85, 40), (70, 40), (60, 20), (55, 10), (45, 10)]

/-- The total number of students -/
def totalStudents : ℕ := 150

/-- Theorem stating that the average score is 75.33 (11300/150) -/
theorem average_score_is_correct :
  averageScore scoreDistribution = 11300 / 150 := by
  sorry

/-- Theorem verifying the total number of students -/
theorem total_students_is_correct :
  (scoreDistribution.foldl (fun acc (_, count) => acc + count) 0) = totalStudents := by
  sorry

end average_score_is_correct_total_students_is_correct_l1364_136413


namespace max_ladles_l1364_136431

/-- Represents the cost of a pan in dollars -/
def pan_cost : ℕ := 3

/-- Represents the cost of a pot in dollars -/
def pot_cost : ℕ := 5

/-- Represents the cost of a ladle in dollars -/
def ladle_cost : ℕ := 9

/-- Represents the total amount Sarah will spend in dollars -/
def total_spend : ℕ := 100

/-- Represents the minimum number of each item Sarah must buy -/
def min_items : ℕ := 2

theorem max_ladles :
  ∃ (p q l : ℕ),
    p ≥ min_items ∧
    q ≥ min_items ∧
    l ≥ min_items ∧
    pan_cost * p + pot_cost * q + ladle_cost * l = total_spend ∧
    l = 9 ∧
    ∀ (p' q' l' : ℕ),
      p' ≥ min_items →
      q' ≥ min_items →
      l' ≥ min_items →
      pan_cost * p' + pot_cost * q' + ladle_cost * l' = total_spend →
      l' ≤ l :=
by sorry

end max_ladles_l1364_136431


namespace max_band_members_l1364_136467

theorem max_band_members : ∃ (m : ℕ), m = 234 ∧
  (∃ (k : ℕ), m = k^2 + 9) ∧
  (∃ (n : ℕ), m = n * (n + 5)) ∧
  (∀ (m' : ℕ), m' > m →
    (∃ (k : ℕ), m' = k^2 + 9) →
    (∃ (n : ℕ), m' = n * (n + 5)) →
    False) :=
by sorry

end max_band_members_l1364_136467


namespace sum_of_terms_l1364_136443

theorem sum_of_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, S n = n^2 + n + 1) →
  (∀ n : ℕ, S (n + 1) - S n = a (n + 1)) →
  a 8 + a 9 + a 10 + a 11 + a 12 = 100 := by
sorry

end sum_of_terms_l1364_136443


namespace geometric_sequence_term_count_l1364_136460

theorem geometric_sequence_term_count :
  ∀ (a : ℕ → ℚ),
  (∀ k : ℕ, a (k + 1) = a k * (1/2)) →  -- Geometric sequence with q = 1/2
  a 1 = 1/2 →                           -- First term a₁ = 1/2
  (∃ n : ℕ, a n = 1/32) →               -- Some term aₙ = 1/32
  ∃ n : ℕ, n = 5 ∧ a n = 1/32 :=        -- The term count n is 5
by sorry

end geometric_sequence_term_count_l1364_136460


namespace pentagon_lcm_problem_l1364_136422

/-- Given five distinct natural numbers on the vertices of a pentagon,
    if the LCM of each pair of adjacent numbers is the same for all sides,
    then the smallest possible value for this common LCM is 30. -/
theorem pentagon_lcm_problem (a b c d e : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e →
  ∃ L : ℕ, L > 0 ∧
    Nat.lcm a b = L ∧
    Nat.lcm b c = L ∧
    Nat.lcm c d = L ∧
    Nat.lcm d e = L ∧
    Nat.lcm e a = L →
  (∀ M : ℕ, M > 0 ∧
    (∃ x y z w v : ℕ, x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧ y ≠ z ∧ y ≠ w ∧ y ≠ v ∧ z ≠ w ∧ z ≠ v ∧ w ≠ v ∧
      Nat.lcm x y = M ∧
      Nat.lcm y z = M ∧
      Nat.lcm z w = M ∧
      Nat.lcm w v = M ∧
      Nat.lcm v x = M) →
    M ≥ 30) :=
by sorry

end pentagon_lcm_problem_l1364_136422


namespace sufficient_not_necessary_parallel_l1364_136452

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- Given vectors a = (m,1) and b = (n,1), m/n = 1 is a sufficient but not necessary condition for a ∥ b -/
theorem sufficient_not_necessary_parallel (m n : ℝ) :
  (m / n = 1 → parallel (m, 1) (n, 1)) ∧
  ¬(parallel (m, 1) (n, 1) → m / n = 1) :=
sorry

end sufficient_not_necessary_parallel_l1364_136452


namespace number_of_proper_subsets_of_union_l1364_136445

def A : Finset Nat := {2, 3}
def B : Finset Nat := {2, 4, 5}

theorem number_of_proper_subsets_of_union : (Finset.powerset (A ∪ B)).card - 1 = 15 := by
  sorry

end number_of_proper_subsets_of_union_l1364_136445


namespace martin_bell_ringing_l1364_136463

theorem martin_bell_ringing (small big : ℕ) : 
  small = (big / 3) + 4 →  -- Condition 1
  small + big = 52 →      -- Condition 2
  big = 36 :=             -- Conclusion
by sorry

end martin_bell_ringing_l1364_136463


namespace arithmetic_sequence_max_sum_l1364_136410

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmeticSum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem arithmetic_sequence_max_sum :
  let a₁ : ℚ := 4
  let d : ℚ := -5/7
  ∀ n : ℕ, n ≠ 0 → arithmeticSum a₁ d 6 ≥ arithmeticSum a₁ d n :=
sorry

end arithmetic_sequence_max_sum_l1364_136410


namespace expression_evaluation_l1364_136481

theorem expression_evaluation : 12 - 10 + 8 * 7 + 6 - 5 * 4 + 3 / 3 - 2 = 43 := by
  sorry

end expression_evaluation_l1364_136481


namespace complex_product_theorem_l1364_136478

theorem complex_product_theorem (z₁ z₂ : ℂ) : 
  (z₁.re = 1 ∧ z₁.im = 1) → (z₂.re = 1 ∧ z₂.im = -1) → z₁ * z₂ = 2 := by
  sorry

end complex_product_theorem_l1364_136478


namespace max_value_of_g_l1364_136480

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + c * x + 3

def tangent_perpendicular (c : ℝ) : Prop :=
  (deriv (f c) 0) * 1 = -1

noncomputable def g (c : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - deriv (f c) x

theorem max_value_of_g (c : ℝ) :
  tangent_perpendicular c →
  ∃ (x_max : ℝ), x_max > 0 ∧ g c x_max = 2 * Real.log 2 - 1 ∧
  ∀ (x : ℝ), x > 0 → g c x ≤ g c x_max :=
sorry

end max_value_of_g_l1364_136480


namespace remaining_amount_is_correct_l1364_136426

-- Define the problem parameters
def initial_amount : ℚ := 100
def action_figure_quantity : ℕ := 3
def board_game_quantity : ℕ := 2
def puzzle_set_quantity : ℕ := 4
def action_figure_price : ℚ := 12
def board_game_price : ℚ := 11
def puzzle_set_price : ℚ := 6
def action_figure_discount : ℚ := 0.25
def sales_tax_rate : ℚ := 0.05

-- Define the function to calculate the remaining amount
def calculate_remaining_amount : ℚ :=
  let discounted_action_figure_price := action_figure_price * (1 - action_figure_discount)
  let action_figure_total := discounted_action_figure_price * action_figure_quantity
  let board_game_total := board_game_price * board_game_quantity
  let puzzle_set_total := puzzle_set_price * puzzle_set_quantity
  let subtotal := action_figure_total + board_game_total + puzzle_set_total
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  initial_amount - total_with_tax

-- Theorem statement
theorem remaining_amount_is_correct :
  calculate_remaining_amount = 23.35 := by sorry

end remaining_amount_is_correct_l1364_136426


namespace rectangle_area_l1364_136466

/-- Proves that the area of a rectangle is 108 square inches, given that its length is 3 times its width and its width is 6 inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) : 
  width = 6 → length = 3 * width → width * length = 108 := by
  sorry

end rectangle_area_l1364_136466


namespace intersection_has_two_elements_l1364_136484

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = p.1^2}
def B : Set (ℝ × ℝ) := {p | p.2 = 1 - |p.1|}

-- State the theorem
theorem intersection_has_two_elements :
  ∃ (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ A ∩ B = {p₁, p₂} :=
sorry

end intersection_has_two_elements_l1364_136484


namespace mn_positive_necessary_mn_positive_not_sufficient_l1364_136427

/-- Definition of an ellipse equation -/
def is_ellipse_equation (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / m + y^2 / n = 1 ∧ m ≠ n ∧ m > 0 ∧ n > 0

/-- The condition mn > 0 is necessary for the equation to represent an ellipse -/
theorem mn_positive_necessary (m n : ℝ) :
  is_ellipse_equation m n → m * n > 0 :=
sorry

/-- The condition mn > 0 is not sufficient for the equation to represent an ellipse -/
theorem mn_positive_not_sufficient :
  ∃ (m n : ℝ), m * n > 0 ∧ ¬(is_ellipse_equation m n) :=
sorry

end mn_positive_necessary_mn_positive_not_sufficient_l1364_136427


namespace cookie_recipe_total_cups_l1364_136498

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the cups of sugar used -/
def totalCups (ratio : RecipeRatio) (sugarCups : ℕ) : ℕ :=
  let partSize := sugarCups / ratio.sugar
  (ratio.butter + ratio.flour + ratio.sugar) * partSize

/-- Theorem stating that for the given recipe ratio and sugar amount, the total cups is 18 -/
theorem cookie_recipe_total_cups :
  let ratio := RecipeRatio.mk 1 2 3
  let sugarCups := 9
  totalCups ratio sugarCups = 18 := by
  sorry

#check cookie_recipe_total_cups

end cookie_recipe_total_cups_l1364_136498


namespace djibos_sister_age_l1364_136485

/-- Given that Djibo is 17 years old and 5 years ago the sum of his and his sister's ages was 35,
    prove that his sister is 28 years old today. -/
theorem djibos_sister_age :
  ∀ (djibo_age sister_age : ℕ),
    djibo_age = 17 →
    djibo_age + sister_age = 35 + 5 →
    sister_age = 28 :=
by sorry

end djibos_sister_age_l1364_136485


namespace sum_and_square_difference_l1364_136417

theorem sum_and_square_difference (x y : ℝ) : 
  x + y = 15 → x^2 - y^2 = 150 → x - y = 10 := by
sorry

end sum_and_square_difference_l1364_136417


namespace power_sum_equality_l1364_136487

theorem power_sum_equality : 3^3 + 4^3 + 5^3 = 6^3 := by
  sorry

end power_sum_equality_l1364_136487


namespace factor_grid_theorem_l1364_136449

/-- The factors of 100 -/
def factors_of_100 : Finset Nat := {1, 2, 4, 5, 10, 20, 25, 50, 100}

/-- The product of all factors of 100 -/
def product_of_factors : Nat := Finset.prod factors_of_100 id

/-- The common product for each row, column, and diagonal -/
def common_product : Nat := 1000

/-- The 3x3 grid representation -/
structure Grid :=
  (a b c d e f g h i : Nat)

/-- Predicate to check if a grid is valid -/
def is_valid_grid (grid : Grid) : Prop :=
  grid.a ∈ factors_of_100 ∧ grid.b ∈ factors_of_100 ∧ grid.c ∈ factors_of_100 ∧
  grid.d ∈ factors_of_100 ∧ grid.e ∈ factors_of_100 ∧ grid.f ∈ factors_of_100 ∧
  grid.g ∈ factors_of_100 ∧ grid.h ∈ factors_of_100 ∧ grid.i ∈ factors_of_100

/-- Predicate to check if a grid satisfies the product condition -/
def satisfies_product_condition (grid : Grid) : Prop :=
  grid.a * grid.b * grid.c = common_product ∧
  grid.d * grid.e * grid.f = common_product ∧
  grid.g * grid.h * grid.i = common_product ∧
  grid.a * grid.d * grid.g = common_product ∧
  grid.b * grid.e * grid.h = common_product ∧
  grid.c * grid.f * grid.i = common_product ∧
  grid.a * grid.e * grid.i = common_product ∧
  grid.c * grid.e * grid.g = common_product

/-- The main theorem -/
theorem factor_grid_theorem (x : Nat) :
  is_valid_grid { a := x, b := 1, c := 50, d := 2, e := 25, f := 20, g := 10, h := 4, i := 5 } ∧
  satisfies_product_condition { a := x, b := 1, c := 50, d := 2, e := 25, f := 20, g := 10, h := 4, i := 5 } →
  x = 20 := by
  sorry

end factor_grid_theorem_l1364_136449


namespace yellow_green_difference_l1364_136423

/-- The number of buttons purchased by a tailor -/
def total_buttons : ℕ := 275

/-- The number of green buttons purchased -/
def green_buttons : ℕ := 90

/-- The number of blue buttons purchased -/
def blue_buttons : ℕ := green_buttons - 5

/-- The number of yellow buttons purchased -/
def yellow_buttons : ℕ := total_buttons - green_buttons - blue_buttons

/-- Theorem stating the difference between yellow and green buttons -/
theorem yellow_green_difference : 
  yellow_buttons - green_buttons = 10 := by sorry

end yellow_green_difference_l1364_136423


namespace sum_a_c_equals_six_l1364_136419

theorem sum_a_c_equals_six (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 36) 
  (h2 : b + d = 6) : 
  a + c = 6 := by
sorry

end sum_a_c_equals_six_l1364_136419


namespace blue_hat_cost_l1364_136408

/-- Proves that the cost of each blue hat is $6 given the conditions of the hat purchase problem -/
theorem blue_hat_cost (total_hats : ℕ) (green_hats : ℕ) (green_cost : ℕ) (total_cost : ℕ) : 
  total_hats = 85 →
  green_hats = 40 →
  green_cost = 7 →
  total_cost = 550 →
  (total_cost - green_hats * green_cost) / (total_hats - green_hats) = 6 := by
  sorry

end blue_hat_cost_l1364_136408


namespace roots_difference_squared_l1364_136440

theorem roots_difference_squared (α β : ℝ) : 
  α^2 - 3*α + 2 = 0 → β^2 - 3*β + 2 = 0 → α ≠ β → (α - β)^2 = 1 := by
  sorry

end roots_difference_squared_l1364_136440


namespace hyperbola_vertex_distance_l1364_136400

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  16 * x^2 - 64 * x - 4 * y^2 + 8 * y + 60 = 0

/-- The distance between the vertices of the hyperbola -/
def vertex_distance (h : ∃ x y, hyperbola_equation x y) : ℝ :=
  1

theorem hyperbola_vertex_distance :
  ∀ h : ∃ x y, hyperbola_equation x y,
  vertex_distance h = 1 := by
sorry

end hyperbola_vertex_distance_l1364_136400


namespace cubic_equation_roots_l1364_136459

theorem cubic_equation_roots (k m : ℝ) : 
  (∃ a b c : ℤ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ x : ℝ, x^3 - 8*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  k + m = 27 := by sorry

end cubic_equation_roots_l1364_136459


namespace average_pages_of_books_l1364_136477

theorem average_pages_of_books (books : List ℕ) (h : books = [120, 150, 180, 210, 240]) : 
  (books.sum / books.length : ℚ) = 180 := by
  sorry

end average_pages_of_books_l1364_136477


namespace tank_fill_level_l1364_136411

theorem tank_fill_level (tank_capacity : ℚ) (added_amount : ℚ) (final_fraction : ℚ) 
  (h1 : tank_capacity = 42)
  (h2 : added_amount = 7)
  (h3 : final_fraction = 9/10)
  (h4 : (final_fraction * tank_capacity) = (added_amount + (initial_fraction * tank_capacity))) :
  initial_fraction = 733/1000 := by
  sorry

end tank_fill_level_l1364_136411


namespace quadratic_factorization_l1364_136402

theorem quadratic_factorization (C D : ℤ) :
  (∀ y, 15 * y^2 - 82 * y + 56 = (C * y - 14) * (D * y - 4)) →
  C * D + C = 20 := by
sorry

end quadratic_factorization_l1364_136402


namespace acidic_concentration_after_water_removal_l1364_136457

/-- Calculates the final concentration of an acidic solution after removing water -/
theorem acidic_concentration_after_water_removal
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (water_removed : ℝ)
  (h1 : initial_volume = 27)
  (h2 : initial_concentration = 0.4)
  (h3 : water_removed = 9)
  : (initial_volume * initial_concentration) / (initial_volume - water_removed) = 0.6 := by
  sorry

#check acidic_concentration_after_water_removal

end acidic_concentration_after_water_removal_l1364_136457


namespace reflection_count_theorem_l1364_136415

/-- Represents a semicircular room -/
structure SemicircularRoom where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a light beam -/
structure LightBeam where
  start : ℝ × ℝ
  angle : ℝ

/-- Counts the number of reflections before the light beam returns to its starting point -/
def count_reflections (room : SemicircularRoom) (beam : LightBeam) : ℕ :=
  sorry

/-- The main theorem stating the number of reflections -/
theorem reflection_count_theorem (room : SemicircularRoom) (beam : LightBeam) :
  room.center = (0, 0) →
  room.radius = 1 →
  beam.start = (-1, 0) →
  beam.angle = 46 * π / 180 →
  count_reflections room beam = 65 :=
sorry

end reflection_count_theorem_l1364_136415
