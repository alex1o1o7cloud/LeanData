import Mathlib

namespace systematic_sample_theorem_l725_72554

/-- Systematic sampling from a population -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  groups : ℕ
  first_group_draw : ℕ
  nth_group_draw : ℕ
  nth_group : ℕ

/-- Theorem for systematic sampling -/
theorem systematic_sample_theorem (s : SystematicSample)
  (h1 : s.population = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.groups = 20)
  (h4 : s.population = s.groups * 8)
  (h5 : s.nth_group_draw = 126)
  (h6 : s.nth_group = 16) :
  s.first_group_draw = 6 := by
  sorry

end systematic_sample_theorem_l725_72554


namespace real_part_of_reciprocal_l725_72573

theorem real_part_of_reciprocal (x y : ℝ) (z : ℂ) (h1 : z = x + y * I) (h2 : z ≠ x) (h3 : Complex.abs z = 1) :
  (1 / (2 - z)).re = (2 - x) / (5 - 4 * x) := by
  sorry

end real_part_of_reciprocal_l725_72573


namespace quadrilateral_diagonal_length_l725_72580

/-- Represents a quadrilateral EFGH with given side lengths -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℝ)
  (EG : ℕ)

/-- The specific quadrilateral from the problem -/
def problem_quadrilateral : Quadrilateral :=
  { EF := 7
  , FG := 21
  , GH := 7
  , HE := 13
  , EG := 21 }

/-- Triangle inequality theorem -/
axiom triangle_inequality {a b c : ℝ} : a + b > c

theorem quadrilateral_diagonal_length : 
  ∀ q : Quadrilateral, 
  q.EF = 7 → q.FG = 21 → q.GH = 7 → q.HE = 13 → 
  q.EG = problem_quadrilateral.EG :=
by
  sorry

#check quadrilateral_diagonal_length

end quadrilateral_diagonal_length_l725_72580


namespace matrix_multiplication_example_l725_72515

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 3]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; -1, 0]
  A * B = !![4, 2; -3, 0] := by sorry

end matrix_multiplication_example_l725_72515


namespace simplify_fraction_l725_72553

theorem simplify_fraction (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ 2*b) :
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end simplify_fraction_l725_72553


namespace min_cone_volume_with_sphere_l725_72561

/-- The minimum volume of a cone that contains a sphere of radius 1 touching its base -/
theorem min_cone_volume_with_sphere (r : ℝ) (h : r = 1) : 
  ∃ (V : ℝ), V = Real.pi * 8 / 3 ∧ 
  (∀ (cone_volume : ℝ), 
    (∃ (R h : ℝ), 
      cone_volume = Real.pi * R^2 * h / 3 ∧ 
      r^2 + (R - r)^2 = h^2) → 
    V ≤ cone_volume) :=
by sorry

end min_cone_volume_with_sphere_l725_72561


namespace condition1_condition2_degree_in_x_l725_72510

/-- A polynomial in three variables -/
def f (x y z : ℝ) : ℝ := (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

/-- The first condition of the polynomial -/
theorem condition1 (x y z : ℝ) : f x (z^2) y + f x (y^2) z = 0 := by sorry

/-- The second condition of the polynomial -/
theorem condition2 (x y z : ℝ) : f (z^3) y x + f (x^3) y z = 0 := by sorry

/-- The polynomial is of 4th degree in x -/
theorem degree_in_x : ∃ (a b c d e : ℝ → ℝ → ℝ), ∀ x y z, 
  f x y z = a y z * x^4 + b y z * x^3 + c y z * x^2 + d y z * x + e y z := by sorry

end condition1_condition2_degree_in_x_l725_72510


namespace not_both_perfect_squares_l725_72598

theorem not_both_perfect_squares (p q : ℕ) (hp : p > 0) (hq : q > 0) :
  ¬(∃ (a b : ℕ), p^2 + q = a^2 ∧ p + q^2 = b^2) := by
  sorry

end not_both_perfect_squares_l725_72598


namespace least_square_value_l725_72519

theorem least_square_value (a x y : ℕ+) 
  (h1 : 15 * a + 165 = x^2)
  (h2 : 16 * a - 155 = y^2) :
  min (x^2) (y^2) ≥ 231361 := by
  sorry

end least_square_value_l725_72519


namespace simple_interest_principle_l725_72511

theorem simple_interest_principle (r t A : ℚ) (h1 : r = 5 / 100) (h2 : t = 12 / 5) (h3 : A = 896) :
  ∃ P : ℚ, P * (1 + r * t) = A ∧ P = 800 := by
  sorry

end simple_interest_principle_l725_72511


namespace min_abs_z_min_abs_z_achievable_l725_72578

open Complex

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 5*I) + Complex.abs (z - 6) = 7) : 
  Complex.abs z ≥ 30 / Real.sqrt 61 := by
  sorry

theorem min_abs_z_achievable : ∃ z : ℂ, 
  (Complex.abs (z - 5*I) + Complex.abs (z - 6) = 7) ∧ 
  (Complex.abs z = 30 / Real.sqrt 61) := by
  sorry

end min_abs_z_min_abs_z_achievable_l725_72578


namespace subset_inequality_l725_72577

-- Define the set S_n
def S_n (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}

-- Define the properties of function f
def is_valid_f (n : ℕ) (f : Set ℕ → ℝ) : Prop :=
  (∀ A : Set ℕ, A ⊆ S_n n → f A > 0) ∧
  (∀ A : Set ℕ, ∀ x y : ℕ, A ⊆ S_n n → x ∈ S_n n → y ∈ S_n n → x ≠ y →
    f (A ∪ {x}) * f (A ∪ {y}) ≤ f (A ∪ {x, y}) * f A)

-- State the theorem
theorem subset_inequality (n : ℕ) (f : Set ℕ → ℝ) (h : is_valid_f n f) :
  ∀ A B : Set ℕ, A ⊆ S_n n → B ⊆ S_n n →
    f A * f B ≤ f (A ∪ B) * f (A ∩ B) :=
sorry

end subset_inequality_l725_72577


namespace blueberry_carton_size_l725_72504

/-- The number of ounces in a carton of blueberries -/
def blueberry_carton_ounces : ℝ := 6

/-- The cost of a carton of blueberries in dollars -/
def blueberry_carton_cost : ℝ := 5

/-- The cost of a carton of raspberries in dollars -/
def raspberry_carton_cost : ℝ := 3

/-- The number of ounces in a carton of raspberries -/
def raspberry_carton_ounces : ℝ := 8

/-- The number of batches of muffins being made -/
def num_batches : ℝ := 4

/-- The number of ounces of fruit required per batch -/
def ounces_per_batch : ℝ := 12

/-- The amount saved by using raspberries instead of blueberries -/
def amount_saved : ℝ := 22

theorem blueberry_carton_size :
  blueberry_carton_ounces = 6 :=
sorry

end blueberry_carton_size_l725_72504


namespace geometric_sequence_characterization_l725_72524

def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_characterization (a : ℕ → ℚ) :
  is_geometric_sequence a ↔ 
  (∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n) :=
sorry

end geometric_sequence_characterization_l725_72524


namespace complex_distance_l725_72590

theorem complex_distance (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs (z₁ + z₂) = 2 * Real.sqrt 2)
  (h₂ : Complex.abs z₁ = Real.sqrt 3)
  (h₃ : Complex.abs z₂ = Real.sqrt 2) :
  Complex.abs (z₁ - z₂) = Real.sqrt 2 := by
  sorry

end complex_distance_l725_72590


namespace equation_solution_l725_72533

theorem equation_solution : ∃! y : ℝ, 5 * y - 100 = 125 ∧ y = 45 := by
  sorry

end equation_solution_l725_72533


namespace square_to_rectangle_ratio_l725_72541

/-- Given a square of side length 4, with E and F as midpoints of opposite sides,
    and AG perpendicular to BF, prove that when dissected into four pieces and
    reassembled into a rectangle, the ratio of height to base is 4/5 -/
theorem square_to_rectangle_ratio (square_side : ℝ) (E F G : ℝ × ℝ) 
  (h1 : square_side = 4)
  (h2 : E.1 = 2 ∧ E.2 = 0)
  (h3 : F.1 = 0 ∧ F.2 = 2)
  (h4 : (G.1 - 4) * (F.2 - 0) = (G.2 - 0) * (F.1 - 4)) -- AG ⟂ BF
  : ∃ (rect_height rect_base : ℝ),
    rect_height * rect_base = square_side^2 ∧
    rect_height / rect_base = 4/5 := by
  sorry

end square_to_rectangle_ratio_l725_72541


namespace perpendicular_vectors_l725_72593

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 3)

theorem perpendicular_vectors (m : ℝ) : 
  (a.1 * (a.1 + m * b.1) + a.2 * (a.2 + m * b.2) = 0) → m = 5 := by
  sorry

end perpendicular_vectors_l725_72593


namespace greatest_of_three_consecutive_integers_l725_72589

theorem greatest_of_three_consecutive_integers (x : ℤ) :
  (x + (x + 1) + (x + 2) = 33) → (max x (max (x + 1) (x + 2)) = 12) :=
by sorry

end greatest_of_three_consecutive_integers_l725_72589


namespace sum_of_N_and_K_is_8_l725_72570

/-- The complex conjugate of a complex number -/
noncomputable def conj (z : ℂ) : ℂ := sorry

/-- The transformation function g -/
noncomputable def g (z : ℂ) : ℂ := 2 * Complex.I * conj z

/-- The polynomial P -/
def P (z : ℂ) : ℂ := z^4 + 6*z^3 + 2*z^2 + 4*z + 1

/-- The roots of P -/
noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def z3 : ℂ := sorry
noncomputable def z4 : ℂ := sorry

/-- The polynomial R -/
noncomputable def R (z : ℂ) : ℂ := z^4 + M*z^3 + N*z^2 + L*z + K
  where
  M : ℂ := sorry
  N : ℂ := sorry
  L : ℂ := sorry
  K : ℂ := sorry

theorem sum_of_N_and_K_is_8 : N + K = 8 := by sorry

end sum_of_N_and_K_is_8_l725_72570


namespace alien_martian_limb_difference_l725_72516

/-- Number of arms an Alien has -/
def alien_arms : ℕ := 3

/-- Number of legs an Alien has -/
def alien_legs : ℕ := 8

/-- Number of arms a Martian has -/
def martian_arms : ℕ := 2 * alien_arms

/-- Number of legs a Martian has -/
def martian_legs : ℕ := alien_legs / 2

/-- Total number of limbs for an Alien -/
def alien_limbs : ℕ := alien_arms + alien_legs

/-- Total number of limbs for a Martian -/
def martian_limbs : ℕ := martian_arms + martian_legs

/-- Number of Aliens and Martians in the comparison -/
def group_size : ℕ := 5

theorem alien_martian_limb_difference :
  group_size * alien_limbs - group_size * martian_limbs = 5 := by
  sorry

end alien_martian_limb_difference_l725_72516


namespace quadratic_no_fixed_points_l725_72556

/-- A quadratic function f(x) = x^2 + ax + 1 has no fixed points if and only if -1 < a < 3 -/
theorem quadratic_no_fixed_points (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 1 ≠ x) ↔ -1 < a ∧ a < 3 := by
  sorry

end quadratic_no_fixed_points_l725_72556


namespace box_dimensions_theorem_l725_72587

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  smallest : ℝ
  middle : ℝ
  largest : ℝ

/-- The conditions given in the problem -/
def satisfiesConditions (d : BoxDimensions) : Prop :=
  d.smallest + d.largest = 17 ∧
  d.smallest + d.middle = 13 ∧
  d.middle + d.largest = 20

/-- The theorem to prove -/
theorem box_dimensions_theorem (d : BoxDimensions) :
  satisfiesConditions d → d = BoxDimensions.mk 5 8 12 := by
  sorry

end box_dimensions_theorem_l725_72587


namespace monochromatic_square_exists_l725_72543

/-- A color type with two possible values -/
inductive Color
  | Red
  | Blue

/-- A point in the 2D grid -/
structure Point where
  x : Nat
  y : Nat
  h_x : x ≥ 1 ∧ x ≤ 5
  h_y : y ≥ 1 ∧ y ≤ 5

/-- A coloring of the 5x5 grid -/
def Coloring := Point → Color

/-- Check if four points form a square with sides parallel to the axes -/
def isSquare (p1 p2 p3 p4 : Point) : Prop :=
  ∃ k : Nat, k > 0 ∧
    ((p1.x + k = p2.x ∧ p1.y = p2.y ∧
      p2.x = p3.x ∧ p2.y + k = p3.y ∧
      p3.x - k = p4.x ∧ p3.y = p4.y ∧
      p4.x = p1.x ∧ p4.y + k = p1.y) ∨
     (p1.y + k = p2.y ∧ p1.x = p2.x ∧
      p2.y = p3.y ∧ p2.x + k = p3.x ∧
      p3.y - k = p4.y ∧ p3.x = p4.x ∧
      p4.y = p1.y ∧ p4.x + k = p1.x))

/-- The main theorem -/
theorem monochromatic_square_exists (c : Coloring) :
  ∃ p1 p2 p3 p4 : Point,
    isSquare p1 p2 p3 p4 ∧
    (c p1 = c p2 ∧ c p2 = c p3 ∨
     c p1 = c p2 ∧ c p2 = c p4 ∨
     c p1 = c p3 ∧ c p3 = c p4 ∨
     c p2 = c p3 ∧ c p3 = c p4) := by
  sorry

end monochromatic_square_exists_l725_72543


namespace function_properties_l725_72588

def f (a : ℝ) (x : ℝ) : ℝ := a * x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

theorem function_properties (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) ∧
  (∀ x, g a (-x) = g a x) ∧
  (∀ x, f a x + g a x = x^2 + a*x + a) ∧
  ((∀ x ∈ Set.Icc 1 2, f a x ≥ 1) ∨ (∃ x ∈ Set.Icc (-1) 2, g a x ≤ -1)) →
  (a ≥ 1 ∨ a ≤ -1) :=
by sorry

end function_properties_l725_72588


namespace largest_consecutive_nonprime_less_than_40_l725_72576

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_consecutive_nonprime_less_than_40 
  (a b c d e : ℕ) 
  (h1 : a + 1 = b)
  (h2 : b + 1 = c)
  (h3 : c + 1 = d)
  (h4 : d + 1 = e)
  (h5 : 10 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 40)
  (h6 : ¬is_prime a ∧ ¬is_prime b ∧ ¬is_prime c ∧ ¬is_prime d ∧ ¬is_prime e) :
  e = 36 :=
sorry

end largest_consecutive_nonprime_less_than_40_l725_72576


namespace range_of_m_l725_72552

-- Define the propositions P and Q
def P (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 + y^2 - 2*m*x + 2*m^2 - 2*m = 0

def Q (m : ℝ) : Prop := 
  let e := Real.sqrt (1 + m/5)
  1 < e ∧ e < 2

-- State the theorem
theorem range_of_m : 
  ∀ m : ℝ, (¬(P m) ∧ Q m) → 2 ≤ m ∧ m < 15 :=
by sorry

end range_of_m_l725_72552


namespace roller_coaster_cars_l725_72513

theorem roller_coaster_cars (people_in_line : ℕ) (people_per_car : ℕ) (num_runs : ℕ) 
  (h1 : people_in_line = 84)
  (h2 : people_per_car = 2)
  (h3 : num_runs = 6)
  (h4 : people_in_line = num_runs * (num_cars * people_per_car)) :
  num_cars = 7 :=
by sorry

end roller_coaster_cars_l725_72513


namespace negation_of_existence_power_of_two_exceeds_1000_l725_72500

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) := by sorry

theorem power_of_two_exceeds_1000 : 
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by sorry

end negation_of_existence_power_of_two_exceeds_1000_l725_72500


namespace darryl_earnings_l725_72550

/-- Calculates the total earnings from selling melons --/
def melon_earnings (
  cantaloupe_price : ℕ)
  (honeydew_price : ℕ)
  (initial_cantaloupes : ℕ)
  (initial_honeydews : ℕ)
  (dropped_cantaloupes : ℕ)
  (rotten_honeydews : ℕ)
  (remaining_cantaloupes : ℕ)
  (remaining_honeydews : ℕ) : ℕ :=
  let sold_cantaloupes := initial_cantaloupes - dropped_cantaloupes - remaining_cantaloupes
  let sold_honeydews := initial_honeydews - rotten_honeydews - remaining_honeydews
  cantaloupe_price * sold_cantaloupes + honeydew_price * sold_honeydews

/-- Theorem stating that Darryl's earnings are $85 --/
theorem darryl_earnings : 
  melon_earnings 2 3 30 27 2 3 8 9 = 85 := by
  sorry

end darryl_earnings_l725_72550


namespace min_value_theorem_l725_72540

theorem min_value_theorem (x y N : ℝ) : 
  (x + 4) * (y - 4) = N → 
  (∀ a b : ℝ, a^2 + b^2 ≥ x^2 + y^2) → 
  x^2 + y^2 = 16 → 
  N = 0 := by
sorry

end min_value_theorem_l725_72540


namespace no_savings_on_joint_purchase_l725_72529

/-- Calculates the cost of windows under a "buy 3, get 1 free" offer -/
def windowCost (regularPrice : ℕ) (quantity : ℕ) : ℕ :=
  ((quantity + 3) / 4 * 3) * regularPrice

/-- Proves that buying windows together does not save money under the given offer -/
theorem no_savings_on_joint_purchase (regularPrice : ℕ) :
  windowCost regularPrice 19 = windowCost regularPrice 9 + windowCost regularPrice 10 :=
by sorry

end no_savings_on_joint_purchase_l725_72529


namespace alices_preferred_numbers_l725_72583

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

def preferred_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 150 ∧ 
  n % 7 = 0 ∧
  ¬(n % 3 = 0) ∧
  is_prime (digit_sum n)

theorem alices_preferred_numbers :
  {n : ℕ | preferred_number n} = {119, 133, 140} := by sorry

end alices_preferred_numbers_l725_72583


namespace cupcake_packages_l725_72567

theorem cupcake_packages (x y z : ℕ) (hx : x = 50) (hy : y = 5) (hz : z = 5) :
  (x - y) / z = 9 := by
  sorry

end cupcake_packages_l725_72567


namespace min_value_theorem_equality_condition_l725_72502

theorem min_value_theorem (a : ℝ) (h : a > 0) : 
  2 * a + 1 / a ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (a : ℝ) (h : a > 0) : 
  (2 * a + 1 / a = 2 * Real.sqrt 2) ↔ (a = Real.sqrt 2 / 2) :=
by sorry

end min_value_theorem_equality_condition_l725_72502


namespace essay_section_ratio_l725_72592

theorem essay_section_ratio (total_words introduction_words body_section_words : ℕ)
  (h1 : total_words = 5000)
  (h2 : introduction_words = 450)
  (h3 : body_section_words = 800)
  (h4 : ∃ (k : ℕ), total_words = introduction_words + 4 * body_section_words + k * introduction_words) :
  ∃ (conclusion_words : ℕ), conclusion_words = 3 * introduction_words :=
by sorry

end essay_section_ratio_l725_72592


namespace max_intersection_points_for_circles_l725_72599

/-- The maximum number of intersection points for n circles in a plane -/
def max_intersection_points (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: Given n circles in a plane, where n ≥ 2, that intersect each other pairwise,
    the maximum number of intersection points is n(n-1). -/
theorem max_intersection_points_for_circles (n : ℕ) (h : n ≥ 2) :
  max_intersection_points n = n * (n - 1) :=
by sorry

end max_intersection_points_for_circles_l725_72599


namespace vacation_cost_l725_72525

theorem vacation_cost (C : ℝ) : 
  (C / 5 - C / 8 = 60) → C = 800 := by sorry

end vacation_cost_l725_72525


namespace tea_boxes_problem_l725_72584

/-- Proves that if there are four boxes of tea, and after removing 9 kg from each box,
    the total remaining quantity equals the original quantity in one box,
    then each box initially contained 12 kg of tea. -/
theorem tea_boxes_problem (x : ℝ) : 
  (4 * (x - 9) = x) → x = 12 := by
  sorry

end tea_boxes_problem_l725_72584


namespace handshake_count_l725_72542

theorem handshake_count (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end handshake_count_l725_72542


namespace absent_fraction_proof_l725_72591

/-- Proves that if work increases by 1/6 when a fraction of members are absent,
    then the fraction of absent members is 1/7 -/
theorem absent_fraction_proof (p : ℕ) (p_pos : p > 0) :
  let increase_factor : ℚ := 1 / 6
  let absent_fraction : ℚ := 1 / 7
  (1 : ℚ) + increase_factor = 1 / (1 - absent_fraction) :=
by sorry

end absent_fraction_proof_l725_72591


namespace cricket_bat_selling_price_l725_72507

-- Define the profit
def profit : ℝ := 150

-- Define the profit percentage
def profitPercentage : ℝ := 20

-- Define the selling price
def sellingPrice : ℝ := 900

-- Theorem to prove
theorem cricket_bat_selling_price :
  let costPrice := profit / (profitPercentage / 100)
  sellingPrice = costPrice + profit := by
  sorry

end cricket_bat_selling_price_l725_72507


namespace abs_neg_two_equals_two_l725_72538

theorem abs_neg_two_equals_two : |-2| = 2 := by
  sorry

end abs_neg_two_equals_two_l725_72538


namespace least_five_digit_square_and_cube_l725_72575

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem least_five_digit_square_and_cube : 
  (is_five_digit 15625 ∧ is_perfect_square 15625 ∧ is_perfect_cube 15625) ∧ 
  (∀ n : ℕ, n < 15625 → ¬(is_five_digit n ∧ is_perfect_square n ∧ is_perfect_cube n)) :=
by sorry

end least_five_digit_square_and_cube_l725_72575


namespace arithmetic_sequence_problem_l725_72522

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence with a_3 = 2 and a_6 = 5, prove a_9 = 8 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_3 : a 3 = 2) 
    (h_6 : a 6 = 5) : 
  a 9 = 8 := by
sorry

end arithmetic_sequence_problem_l725_72522


namespace problem_solution_l725_72560

theorem problem_solution (x y z : ℝ) 
  (h1 : x*z/(x+y) + y*z/(y+z) + x*y/(z+x) = -18)
  (h2 : z*y/(x+y) + z*x/(y+z) + y*x/(z+x) = 20) :
  y/(x+y) + z/(y+z) + x/(z+x) = 20.5 := by
  sorry

end problem_solution_l725_72560


namespace cody_marbles_l725_72537

theorem cody_marbles (initial_marbles : ℕ) : 
  (initial_marbles - initial_marbles / 3 - 5 = 7) → initial_marbles = 18 := by
  sorry

end cody_marbles_l725_72537


namespace prime_sum_product_l725_72531

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_sum_product :
  ∃ p q : ℕ,
    is_prime p ∧
    is_prime q ∧
    p + q = 102 ∧
    (p > 30 ∨ q > 30) ∧
    p * q = 2201 :=
by sorry

end prime_sum_product_l725_72531


namespace rectangle_area_equals_perimeter_l725_72517

theorem rectangle_area_equals_perimeter (x : ℝ) : 
  (3 * x) * (x + 5) = 2 * (3 * x) + 2 * (x + 5) → x = 1 := by
  sorry

end rectangle_area_equals_perimeter_l725_72517


namespace simple_interest_rate_calculation_l725_72555

theorem simple_interest_rate_calculation (principal : ℝ) (h : principal > 0) :
  let final_amount := (7 / 6 : ℝ) * principal
  let time := 4
  let interest := final_amount - principal
  let rate := (interest / (principal * time)) * 100
  rate = 100 / 24 := by sorry

end simple_interest_rate_calculation_l725_72555


namespace smallest_even_cube_ending_392_l725_72532

theorem smallest_even_cube_ending_392 :
  ∀ n : ℕ, n > 0 → Even n → n^3 ≡ 392 [ZMOD 1000] → n ≥ 892 :=
by sorry

end smallest_even_cube_ending_392_l725_72532


namespace computation_proof_l725_72595

theorem computation_proof : 18 * (216 / 3 + 36 / 6 + 4 / 9 + 2 + 1 / 18) = 1449 := by
  sorry

end computation_proof_l725_72595


namespace intersection_of_sets_l725_72546

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3, 4, 5}
  let B : Set ℕ := {1, 2, 4, 6}
  A ∩ B = {1, 2, 4} := by
sorry

end intersection_of_sets_l725_72546


namespace triangle_area_proof_l725_72526

/-- The area of a triangular region bounded by the x-axis, y-axis, and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the line bounding the triangular region -/
def lineEquation (x y : ℝ) : Prop := 3 * x + 2 * y = 12

/-- The x-intercept of the line -/
def xIntercept : ℝ := 4

/-- The y-intercept of the line -/
def yIntercept : ℝ := 6

theorem triangle_area_proof :
  lineEquation xIntercept 0 ∧
  lineEquation 0 yIntercept ∧
  triangleArea = (1 / 2) * xIntercept * yIntercept :=
by
  sorry

end triangle_area_proof_l725_72526


namespace a_range_l725_72508

-- Define the propositions and variables
variable (p q : Prop)
variable (x a : ℝ)

-- Define the conditions
axiom x_range : 1/2 ≤ x ∧ x ≤ 1
axiom q_def : q ↔ (x - a) * (x - a - 1) ≤ 0
axiom not_p_necessary : ¬q → ¬p
axiom not_p_not_sufficient : ¬(¬p → ¬q)

-- State the theorem
theorem a_range : 0 ≤ a ∧ a ≤ 1/2 := by sorry

end a_range_l725_72508


namespace pencil_distribution_l725_72585

theorem pencil_distribution (total_pencils : ℕ) (pencils_per_row : ℕ) (rows : ℕ) : 
  total_pencils = 12 → 
  pencils_per_row = 4 → 
  total_pencils = rows * pencils_per_row → 
  rows = 3 := by
  sorry

end pencil_distribution_l725_72585


namespace jackson_gpa_probability_l725_72509

-- Define the point values for each grade
def pointA : ℚ := 5
def pointB : ℚ := 4
def pointC : ℚ := 2
def pointD : ℚ := 1

-- Define the probabilities for Literature grades
def litProbA : ℚ := 1/5
def litProbB : ℚ := 2/5
def litProbC : ℚ := 2/5

-- Define the probabilities for Sociology grades
def socProbA : ℚ := 1/3
def socProbB : ℚ := 1/2
def socProbC : ℚ := 1/6

-- Define the number of classes
def numClasses : ℕ := 5

-- Define the minimum GPA required
def minGPA : ℚ := 4

-- Define the function to calculate GPA
def calculateGPA (points : ℚ) : ℚ := points / numClasses

-- Theorem statement
theorem jackson_gpa_probability :
  let confirmedPoints : ℚ := pointA + pointA  -- Calculus and Physics
  let minRemainingPoints : ℚ := minGPA * numClasses - confirmedPoints
  let probTwoAs : ℚ := litProbA * socProbA
  let probALitBSoc : ℚ := litProbA * socProbB
  let probASocBLit : ℚ := socProbA * litProbB
  (probTwoAs + probALitBSoc + probASocBLit) = 2/5 := by sorry

end jackson_gpa_probability_l725_72509


namespace volume_maximized_at_10cm_l725_72557

/-- Represents the dimensions of the original sheet --/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the volume of the container given sheet dimensions and cut length --/
def containerVolume (sheet : SheetDimensions) (cutLength : ℝ) : ℝ :=
  (sheet.length - 2 * cutLength) * (sheet.width - 2 * cutLength) * cutLength

/-- Theorem stating that the volume is maximized when cut length is 10cm --/
theorem volume_maximized_at_10cm (sheet : SheetDimensions) 
  (h1 : sheet.length = 90)
  (h2 : sheet.width = 48) :
  ∃ (maxCutLength : ℝ), maxCutLength = 10 ∧ 
  ∀ (x : ℝ), 0 < x → x < sheet.width / 2 → x < sheet.length / 2 → 
  containerVolume sheet x ≤ containerVolume sheet maxCutLength :=
sorry

end volume_maximized_at_10cm_l725_72557


namespace monotonic_decreasing_condition_l725_72528

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 4

-- State the theorem
theorem monotonic_decreasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 6 → f a x₁ > f a x₂) ↔ 0 ≤ a ∧ a ≤ 1/4 :=
by sorry

end monotonic_decreasing_condition_l725_72528


namespace circle_radius_constant_l725_72521

theorem circle_radius_constant (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 2*y + c = 0 ↔ (x + 4)^2 + (y + 1)^2 = 5^2) → 
  c = 42 :=
by sorry

end circle_radius_constant_l725_72521


namespace congruence_problem_l725_72547

theorem congruence_problem (x : ℤ) : 
  (4 * x + 9) % 19 = 3 → (3 * x + 8) % 19 = 13 := by
sorry

end congruence_problem_l725_72547


namespace project_B_highest_score_l725_72527

structure Project where
  name : String
  innovation : ℝ
  practicality : ℝ

def totalScore (p : Project) : ℝ :=
  0.6 * p.innovation + 0.4 * p.practicality

def projectA : Project := ⟨"A", 90, 90⟩
def projectB : Project := ⟨"B", 95, 90⟩
def projectC : Project := ⟨"C", 90, 95⟩
def projectD : Project := ⟨"D", 90, 85⟩

def projects : List Project := [projectA, projectB, projectC, projectD]

theorem project_B_highest_score :
  ∀ p ∈ projects, p ≠ projectB → totalScore p ≤ totalScore projectB :=
sorry

end project_B_highest_score_l725_72527


namespace exist_four_lines_eight_regions_l725_72520

/-- A line in the coordinate plane defined by y = kx + b --/
structure Line where
  k : ℕ
  b : ℕ
  k_in_range : k ∈ Finset.range 9 \ {0}
  b_in_range : b ∈ Finset.range 9 \ {0}

/-- The set of four lines --/
def FourLines : Type := Fin 4 → Line

/-- All coefficients and constants are distinct --/
def all_distinct (lines : FourLines) : Prop :=
  ∀ i j, i ≠ j → lines i ≠ lines j

/-- The number of regions formed by the lines --/
def num_regions (lines : FourLines) : ℕ := sorry

/-- Theorem: There exist 4 lines that divide the plane into 8 regions --/
theorem exist_four_lines_eight_regions :
  ∃ (lines : FourLines), all_distinct lines ∧ num_regions lines = 8 := by sorry

end exist_four_lines_eight_regions_l725_72520


namespace sold_to_production_ratio_l725_72549

def last_year_production : ℕ := 5000
def this_year_production : ℕ := 2 * last_year_production
def phones_left : ℕ := 7500

def sold_phones : ℕ := this_year_production - phones_left

theorem sold_to_production_ratio : 
  (sold_phones : ℚ) / this_year_production = 1 / 4 := by sorry

end sold_to_production_ratio_l725_72549


namespace equal_angle_slope_value_l725_72545

/-- The slope of a line that forms equal angles with y = x and y = 2x --/
def equal_angle_slope : ℝ → Prop := λ k =>
  let l₁ : ℝ → ℝ := λ x => x
  let l₂ : ℝ → ℝ := λ x => 2 * x
  let angle (m₁ m₂ : ℝ) : ℝ := |((m₂ - m₁) / (1 + m₁ * m₂))|
  (angle k 1 = angle 2 k) ∧ (3 * k^2 - 2 * k - 3 = 0)

/-- The slope of a line that forms equal angles with y = x and y = 2x
    is (1 ± √10) / 3 --/
theorem equal_angle_slope_value :
  ∃ k : ℝ, equal_angle_slope k ∧ (k = (1 + Real.sqrt 10) / 3 ∨ k = (1 - Real.sqrt 10) / 3) :=
sorry

end equal_angle_slope_value_l725_72545


namespace cow_increase_is_24_l725_72569

/-- Represents the number of cows at different stages --/
structure CowCount where
  initial : Nat
  after_deaths : Nat
  after_sales : Nat
  current : Nat

/-- Calculates the increase in cow count given the initial conditions and final count --/
def calculate_increase (c : CowCount) (bought : Nat) (gifted : Nat) : Nat :=
  c.current - (c.after_sales + bought + gifted)

/-- Theorem stating that the increase in cows is 24 given the problem conditions --/
theorem cow_increase_is_24 :
  let c := CowCount.mk 39 (39 - 25) ((39 - 25) - 6) 83
  let bought := 43
  let gifted := 8
  calculate_increase c bought gifted = 24 := by
  sorry

end cow_increase_is_24_l725_72569


namespace sum_of_fourth_powers_l725_72505

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_eq : a + b + c = 8)
  (sum_prod_eq : a * b + a * c + b * c = 13)
  (prod_eq : a * b * c = -22) :
  a^4 + b^4 + c^4 = 1378 := by
  sorry

end sum_of_fourth_powers_l725_72505


namespace unique_modular_congruence_l725_72565

theorem unique_modular_congruence : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] := by
  sorry

end unique_modular_congruence_l725_72565


namespace inequality_solution_set_l725_72518

theorem inequality_solution_set : 
  {x : ℝ | 3 ≤ |2 - x| ∧ |2 - x| < 9} = {x : ℝ | -7 < x ∧ x ≤ -1 ∨ 5 ≤ x ∧ x < 11} := by
  sorry

end inequality_solution_set_l725_72518


namespace boat_speed_problem_l725_72501

/-- Proves that given a boat traveling upstream at 3 km/h and having an average
    round-trip speed of 4.2 km/h, its downstream speed is 7 km/h. -/
theorem boat_speed_problem (upstream_speed downstream_speed average_speed : ℝ) 
    (h1 : upstream_speed = 3)
    (h2 : average_speed = 4.2)
    (h3 : average_speed = (2 * upstream_speed * downstream_speed) / (upstream_speed + downstream_speed)) :
  downstream_speed = 7 := by
  sorry

end boat_speed_problem_l725_72501


namespace circular_tank_properties_l725_72536

theorem circular_tank_properties (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) :
  let r := (AB / 2)^2 + DC^2
  (π * r = 244 * π) ∧ (2 * π * Real.sqrt r = 2 * π * Real.sqrt 244) := by
  sorry

end circular_tank_properties_l725_72536


namespace line_through_P_intersecting_C_l725_72582

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 8*y - 5 = 0

-- Define the point P
def point_P : ℝ × ℝ := (5, 0)

-- Define the chord length
def chord_length : ℝ := 8

-- Define the two possible line equations
def line_eq1 (x : ℝ) : Prop := x = 5
def line_eq2 (x y : ℝ) : Prop := 7*x + 24*y - 35 = 0

-- Theorem statement
theorem line_through_P_intersecting_C :
  ∃ (l : ℝ → ℝ → Prop),
    (∀ x y, l x y → (x = point_P.1 ∧ y = point_P.2)) ∧
    (∃ x1 y1 x2 y2, l x1 y1 ∧ l x2 y2 ∧ circle_C x1 y1 ∧ circle_C x2 y2 ∧
      (x2 - x1)^2 + (y2 - y1)^2 = chord_length^2) ∧
    ((∀ x y, l x y ↔ line_eq1 x) ∨ (∀ x y, l x y ↔ line_eq2 x y)) :=
sorry

end line_through_P_intersecting_C_l725_72582


namespace range_of_c_l725_72530

/-- Given c > 0, if the function y = c^x is monotonically decreasing on ℝ or 
    the function g(x) = lg(2cx^2 + 2x + 1) has domain ℝ, but not both, 
    then c ≥ 1 or 0 < c ≤ 1/2 -/
theorem range_of_c (c : ℝ) (h_c : c > 0) : 
  (∀ x y : ℝ, x < y → c^x > c^y) ∨ 
  (∀ x : ℝ, 2*c*x^2 + 2*x + 1 > 0) ∧ 
  ¬((∀ x y : ℝ, x < y → c^x > c^y) ∧ 
    (∀ x : ℝ, 2*c*x^2 + 2*x + 1 > 0)) → 
  c ≥ 1 ∨ (0 < c ∧ c ≤ 1/2) := by
  sorry

end range_of_c_l725_72530


namespace basketball_team_combinations_l725_72564

/-- The number of players in the basketball team -/
def total_players : ℕ := 12

/-- The number of players in the starting lineup (excluding the captain) -/
def starting_lineup : ℕ := 5

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The number of ways to select 1 captain from 12 players and then 5 players 
    from the remaining 11 for the starting lineup is equal to 5544 -/
theorem basketball_team_combinations : 
  total_players * choose (total_players - 1) starting_lineup = 5544 := by
  sorry


end basketball_team_combinations_l725_72564


namespace system_solution_l725_72535

theorem system_solution : ∃! (x y : ℝ), 
  (x + Real.sqrt (x + 2*y) - 2*y = 7/2) ∧ 
  (x^2 + x + 2*y - 4*y^2 = 27/2) ∧
  (x = 19/4) ∧ (y = 17/8) := by
  sorry

end system_solution_l725_72535


namespace max_blue_chips_l725_72558

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_blue_chips 
  (total : ℕ) 
  (red : ℕ) 
  (blue : ℕ) 
  (h_total : total = 72)
  (h_sum : red + blue = total)
  (h_prime : ∃ p : ℕ, is_prime p ∧ red = blue + p) :
  blue ≤ 35 ∧ ∃ blue_max : ℕ, blue_max = 35 ∧ 
    ∃ red_max : ℕ, ∃ p_min : ℕ, 
      is_prime p_min ∧ 
      red_max + blue_max = total ∧ 
      red_max = blue_max + p_min :=
sorry

end max_blue_chips_l725_72558


namespace smallest_n_for_432n_perfect_square_l725_72563

theorem smallest_n_for_432n_perfect_square :
  ∃ (n : ℕ), n > 0 ∧ (∃ (k : ℕ), 432 * n = k^2) ∧
  (∀ (m : ℕ), m > 0 → m < n → ¬∃ (j : ℕ), 432 * m = j^2) ∧
  n = 3 := by
sorry

end smallest_n_for_432n_perfect_square_l725_72563


namespace dennis_marbles_l725_72574

theorem dennis_marbles (laurie kurt dennis : ℕ) 
  (h1 : laurie = kurt + 12)
  (h2 : kurt + 45 = dennis)
  (h3 : laurie = 37) : 
  dennis = 70 := by
sorry

end dennis_marbles_l725_72574


namespace min_value_a_squared_plus_b_squared_l725_72559

theorem min_value_a_squared_plus_b_squared (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → 
  (∀ a' b' : ℝ, (∃ x : ℝ, x^4 + a'*x^3 + b'*x^2 + a'*x + 1 = 0) → a'^2 + b'^2 ≥ 4) ∧
  (∃ a' b' : ℝ, (∃ x : ℝ, x^4 + a'*x^3 + b'*x^2 + a'*x + 1 = 0) ∧ a'^2 + b'^2 = 4) :=
by sorry

end min_value_a_squared_plus_b_squared_l725_72559


namespace diaries_calculation_l725_72594

/-- Calculates the number of diaries after doubling and losing a quarter --/
def diaries_after_change (initial : ℕ) : ℕ :=
  let doubled := initial * 2
  let total := initial + doubled
  total - (total / 4)

/-- Theorem stating that starting with 8 diaries results in 18 after changes --/
theorem diaries_calculation : diaries_after_change 8 = 18 := by
  sorry

end diaries_calculation_l725_72594


namespace B_power_15_minus_3_power_14_l725_72572

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • (B^14) = !![0, 4; 0, -1] := by sorry

end B_power_15_minus_3_power_14_l725_72572


namespace least_positive_integer_for_reducible_fraction_l725_72581

theorem least_positive_integer_for_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), 0 < k ∧ k < n → ¬(∃ (d : ℕ), d > 1 ∧ d ∣ (k - 20) ∧ d ∣ (7 * k + 2))) ∧
  (∃ (d : ℕ), d > 1 ∧ d ∣ (n - 20) ∧ d ∣ (7 * n + 2)) ∧
  n = 22 :=
sorry

end least_positive_integer_for_reducible_fraction_l725_72581


namespace chocolate_bar_problem_l725_72506

/-- Represents the problem of calculating unsold chocolate bars -/
theorem chocolate_bar_problem (cost_per_bar : ℕ) (total_bars : ℕ) (revenue : ℕ) : 
  cost_per_bar = 3 → 
  total_bars = 9 → 
  revenue = 18 → 
  total_bars - (revenue / cost_per_bar) = 3 := by
  sorry

end chocolate_bar_problem_l725_72506


namespace remainder_3_800_mod_17_l725_72514

theorem remainder_3_800_mod_17 : 3^800 % 17 = 1 := by
  sorry

end remainder_3_800_mod_17_l725_72514


namespace roots_imply_p_zero_q_negative_l725_72544

theorem roots_imply_p_zero_q_negative (α β p q : ℝ) : 
  α ≠ β →  -- α and β are distinct
  α^2 + p*α + q = 0 →  -- α is a root of the equation
  β^2 + p*β + q = 0 →  -- β is a root of the equation
  α^3 - α^2*β - α*β^2 + β^3 = 0 →  -- given condition
  p = 0 ∧ q < 0 := by
  sorry

end roots_imply_p_zero_q_negative_l725_72544


namespace rhombus_diagonals_l725_72597

/-- Given a rhombus with area 117 cm² and the perimeter of the rectangle formed by
    the midpoints of its sides is 31 cm, prove that its diagonals are 18 cm and 13 cm. -/
theorem rhombus_diagonals (area : ℝ) (perimeter : ℝ) (d₁ d₂ : ℝ) :
  area = 117 →
  perimeter = 31 →
  d₁ * d₂ / 2 = area →
  d₁ + d₂ = perimeter →
  (d₁ = 18 ∧ d₂ = 13) ∨ (d₁ = 13 ∧ d₂ = 18) := by
  sorry

end rhombus_diagonals_l725_72597


namespace inverse_proportion_comparison_l725_72568

/-- An inverse proportion function passing through (-2, 4) with points (1, y₁) and (3, y₂) on its graph -/
def InverseProportion (k : ℝ) (y₁ y₂ : ℝ) : Prop :=
  k ≠ 0 ∧ 
  4 = k / (-2) ∧ 
  y₁ = k / 1 ∧ 
  y₂ = k / 3

theorem inverse_proportion_comparison (k : ℝ) (y₁ y₂ : ℝ) 
  (h : InverseProportion k y₁ y₂) : y₁ < y₂ := by
  sorry

end inverse_proportion_comparison_l725_72568


namespace first_day_over_500_is_saturday_l725_72596

def days : List String := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

def pens_on_day (day : Nat) : Nat :=
  if day = 0 then 5
  else if day = 1 then 10
  else 10 * (3 ^ (day - 1))

def first_day_over_500 : String :=
  days[(days.findIdx? (fun d => pens_on_day (days.indexOf d) > 500)).getD 0]

theorem first_day_over_500_is_saturday : first_day_over_500 = "Saturday" := by
  sorry

end first_day_over_500_is_saturday_l725_72596


namespace edwards_remaining_money_l725_72512

/-- Calculates the remaining money after a purchase with sales tax -/
def remaining_money (initial_amount purchase_amount tax_rate : ℚ) : ℚ :=
  let sales_tax := purchase_amount * tax_rate
  let total_cost := purchase_amount + sales_tax
  initial_amount - total_cost

/-- Theorem stating that Edward's remaining money is $0.42 -/
theorem edwards_remaining_money :
  remaining_money 18 16.35 (75 / 1000) = 42 / 100 := by
  sorry

end edwards_remaining_money_l725_72512


namespace felix_betty_length_difference_l725_72539

-- Define the given constants
def betty_steps_per_gap : ℕ := 36
def felix_jumps_per_gap : ℕ := 9
def total_posts : ℕ := 51
def total_distance : ℝ := 7920

-- Define the theorem
theorem felix_betty_length_difference :
  let total_gaps := total_posts - 1
  let betty_total_steps := betty_steps_per_gap * total_gaps
  let felix_total_jumps := felix_jumps_per_gap * total_gaps
  let betty_step_length := total_distance / betty_total_steps
  let felix_jump_length := total_distance / felix_total_jumps
  felix_jump_length - betty_step_length = 13.2 := by
sorry

end felix_betty_length_difference_l725_72539


namespace quadratic_equation_a_range_l725_72571

/-- The range of values for a in the quadratic equation (a-1)x^2 + √(a+1)x + 2 = 0 -/
theorem quadratic_equation_a_range :
  ∀ a : ℝ, (∃ x : ℝ, (a - 1) * x^2 + Real.sqrt (a + 1) * x + 2 = 0) →
  (a ≥ -1 ∧ a ≠ 1) :=
by sorry

end quadratic_equation_a_range_l725_72571


namespace table_size_lower_bound_l725_72551

/-- Represents a table with 10 columns and n rows, where each cell contains a digit. -/
structure DigitTable (n : ℕ) :=
  (rows : Fin n → Fin 10 → Fin 10)

/-- 
Given a table with 10 columns and n rows, where each cell contains a digit, 
and for any row A and any two columns, there exists a row that differs from A 
in exactly these two columns, prove that n ≥ 512.
-/
theorem table_size_lower_bound {n : ℕ} (t : DigitTable n) 
  (h : ∀ (A : Fin n) (i j : Fin 10), i ≠ j → 
    ∃ (B : Fin n), (∀ k : Fin 10, k ≠ i ∧ k ≠ j → t.rows A k = t.rows B k) ∧
                   t.rows A i ≠ t.rows B i ∧ 
                   t.rows A j ≠ t.rows B j) : 
  n ≥ 512 := by
  sorry

end table_size_lower_bound_l725_72551


namespace hexagon_reachability_l725_72548

def Hexagon := Fin 6 → ℤ

def initial_hexagon : Hexagon := ![12, 1, 10, 6, 8, 3]

def is_valid_move (h1 h2 : Hexagon) : Prop :=
  ∃ i : Fin 6, 
    (h2 i = h1 i + 1 ∧ h2 ((i + 1) % 6) = h1 ((i + 1) % 6) + 1) ∨
    (h2 i = h1 i - 1 ∧ h2 ((i + 1) % 6) = h1 ((i + 1) % 6) - 1) ∧
    ∀ j : Fin 6, j ≠ i ∧ j ≠ (i + 1) % 6 → h2 j = h1 j

def is_reachable (start goal : Hexagon) : Prop :=
  ∃ (n : ℕ) (sequence : Fin (n + 1) → Hexagon),
    sequence 0 = start ∧
    sequence n = goal ∧
    ∀ i : Fin n, is_valid_move (sequence i) (sequence (i + 1))

theorem hexagon_reachability :
  (is_reachable initial_hexagon ![14, 6, 13, 4, 5, 2]) ∧
  ¬(is_reachable initial_hexagon ![6, 17, 14, 3, 15, 2]) := by
  sorry

end hexagon_reachability_l725_72548


namespace luncheon_invitees_l725_72534

/-- The number of people who didn't show up to the luncheon -/
def no_shows : ℕ := 50

/-- The number of people each table can hold -/
def people_per_table : ℕ := 3

/-- The number of tables needed for the people who showed up -/
def tables_used : ℕ := 6

/-- The total number of people originally invited to the luncheon -/
def total_invited : ℕ := no_shows + people_per_table * tables_used + 1

/-- Theorem stating that the number of people originally invited to the luncheon is 101 -/
theorem luncheon_invitees : total_invited = 101 := by
  sorry

end luncheon_invitees_l725_72534


namespace intersection_of_A_and_B_l725_72566

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | x < 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 ≤ x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l725_72566


namespace largest_angle_in_triangle_l725_72586

theorem largest_angle_in_triangle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α + β = (7/5) * 90 →  -- Sum of two angles is 7/5 of a right angle
  β = α + 40 →  -- One angle is 40° larger than the other
  max α (max β γ) = 83 :=  -- The largest angle is 83°
by sorry

end largest_angle_in_triangle_l725_72586


namespace complex_fraction_equality_l725_72562

theorem complex_fraction_equality : (4 - 2*Complex.I) / (1 + Complex.I) = 1 - 3*Complex.I := by
  sorry

end complex_fraction_equality_l725_72562


namespace shoes_sales_goal_l725_72579

/-- Given a monthly goal and the number of shoes sold in two weeks, 
    calculate the additional pairs needed to meet the goal -/
def additional_pairs_needed (monthly_goal : ℕ) (sold_week1 : ℕ) (sold_week2 : ℕ) : ℕ :=
  monthly_goal - (sold_week1 + sold_week2)

/-- Theorem: Given the specific values from the problem, 
    the additional pairs needed is 41 -/
theorem shoes_sales_goal :
  additional_pairs_needed 80 27 12 = 41 := by
  sorry

end shoes_sales_goal_l725_72579


namespace intersection_M_N_l725_72523

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2 - |x|}

theorem intersection_M_N : M ∩ N = {y : ℝ | 0 ≤ y ∧ y ≤ 2} := by sorry

end intersection_M_N_l725_72523


namespace farmer_land_calculation_l725_72503

/-- Represents the total land owned by the farmer in acres -/
def total_land : ℝ := 7000

/-- Represents the proportion of land that was cleared for planting -/
def cleared_proportion : ℝ := 0.90

/-- Represents the proportion of cleared land planted with potato -/
def potato_proportion : ℝ := 0.20

/-- Represents the proportion of cleared land planted with tomato -/
def tomato_proportion : ℝ := 0.70

/-- Represents the amount of cleared land planted with corn in acres -/
def corn_land : ℝ := 630

theorem farmer_land_calculation :
  total_land * cleared_proportion * (potato_proportion + tomato_proportion) + corn_land = 
  total_land * cleared_proportion := by sorry

end farmer_land_calculation_l725_72503
