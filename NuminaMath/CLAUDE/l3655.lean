import Mathlib

namespace rectangle_side_lengths_l3655_365505

/-- Given a rectangle with one side length b, diagonal length d, and the difference between
    the diagonal and the other side (d-a), prove that the side lengths of the rectangle
    are a = d - √(d² - b²) and b. -/
theorem rectangle_side_lengths
  (b d : ℝ) (h : b > 0) (h' : d > b) :
  let a := d - Real.sqrt (d^2 - b^2)
  (a > 0 ∧ a < d) ∧ 
  (a^2 + b^2 = d^2) ∧
  (d - a = Real.sqrt (d^2 - b^2)) :=
sorry

end rectangle_side_lengths_l3655_365505


namespace function_properties_l3655_365545

open Set

def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
  (h_even : IsEven f)
  (h_periodic : IsPeriodic f 2)
  (h_interval : ∀ x ∈ Icc 1 2, f x = x^2 + 2*x - 1) :
  ∀ x ∈ Icc 0 1, f x = x^2 - 6*x + 7 := by
sorry

end function_properties_l3655_365545


namespace circle_properties_l3655_365526

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*(m+3)*x + 2*(1-4*m^2)*y + 16*m^4 + 9 = 0

-- Define the theorem
theorem circle_properties :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) →
  ((-1/7 < m ∧ m < 1) ∧
   (∃ r : ℝ, 0 < r ∧ r ≤ 4 * Real.sqrt 7 / 7 ∧
    ∀ x y : ℝ, circle_equation x y m → (x - (m+3))^2 + (y - (4*m^2-1))^2 = r^2) ∧
   (∀ y : ℝ, (∃ x : ℝ, circle_equation x y m) → y ≥ -1)) :=
by sorry

end circle_properties_l3655_365526


namespace pipe_cut_theorem_l3655_365577

/-- Given a pipe of length 68 feet cut into two pieces, where one piece is 12 feet shorter than the other, 
    the length of the shorter piece is 28 feet. -/
theorem pipe_cut_theorem : 
  ∀ (shorter_piece longer_piece : ℝ),
  shorter_piece + longer_piece = 68 →
  longer_piece = shorter_piece + 12 →
  shorter_piece = 28 := by
sorry

end pipe_cut_theorem_l3655_365577


namespace solution_for_y_l3655_365559

theorem solution_for_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 1 + 1/y) (eq2 : y = 2 + 1/x) :
  y = 1 + Real.sqrt 3 ∨ y = 1 - Real.sqrt 3 :=
by sorry

end solution_for_y_l3655_365559


namespace complex_plane_properties_l3655_365506

/-- Given complex numbers and their corresponding points in the complex plane, prove various geometric properties. -/
theorem complex_plane_properties (z₁ z₂ z₃ : ℂ) 
  (h₁ : z₁ = -3 + 4*I) 
  (h₂ : z₂ = 1 + 7*I) 
  (h₃ : z₃ = 3 - 4*I) : 
  (z₂.re > 0 ∧ z₂.im > 0) ∧ 
  (z₁ = -z₃) ∧ 
  (z₁.re * z₂.re + z₁.im * z₂.im > 0) ∧
  (z₁.re * (z₂.re + z₃.re) + z₁.im * (z₂.im + z₃.im) = 0) := by
  sorry

end complex_plane_properties_l3655_365506


namespace quadratic_form_sum_l3655_365504

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 := by
  sorry

end quadratic_form_sum_l3655_365504


namespace stamp_collection_difference_l3655_365500

theorem stamp_collection_difference (kylie_stamps nelly_stamps : ℕ) : 
  kylie_stamps = 34 →
  nelly_stamps > kylie_stamps →
  kylie_stamps + nelly_stamps = 112 →
  nelly_stamps - kylie_stamps = 44 := by
  sorry

end stamp_collection_difference_l3655_365500


namespace triangle_side_ratio_sum_bounds_l3655_365519

theorem triangle_side_ratio_sum_bounds (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  1 < a / (b + c) + b / (c + a) + c / (a + b) ∧ a / (b + c) + b / (c + a) + c / (a + b) < 2 := by
  sorry

end triangle_side_ratio_sum_bounds_l3655_365519


namespace tan_fifteen_pi_fourths_l3655_365533

theorem tan_fifteen_pi_fourths : Real.tan (15 * π / 4) = -1 := by sorry

end tan_fifteen_pi_fourths_l3655_365533


namespace volleyball_tournament_triples_l3655_365576

/-- Represents a round-robin volleyball tournament -/
structure Tournament :=
  (num_teams : ℕ)
  (wins_per_team : ℕ)

/-- Represents the number of triples where each team wins once against the others -/
def count_special_triples (t : Tournament) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem volleyball_tournament_triples (t : Tournament) 
  (h1 : t.num_teams = 15)
  (h2 : t.wins_per_team = 7) :
  count_special_triples t = 140 :=
sorry

end volleyball_tournament_triples_l3655_365576


namespace burger_calorie_content_l3655_365537

/-- Represents the calorie content of a lunch meal -/
structure LunchMeal where
  burger_calories : ℕ
  carrot_stick_calories : ℕ
  cookie_calories : ℕ
  carrot_stick_count : ℕ
  cookie_count : ℕ
  total_calories : ℕ

/-- Theorem stating the calorie content of a burger in a specific lunch meal -/
theorem burger_calorie_content (meal : LunchMeal) 
  (h1 : meal.carrot_stick_calories = 20)
  (h2 : meal.cookie_calories = 50)
  (h3 : meal.carrot_stick_count = 5)
  (h4 : meal.cookie_count = 5)
  (h5 : meal.total_calories = 750) :
  meal.burger_calories = 400 := by
  sorry

end burger_calorie_content_l3655_365537


namespace square_rectangle_overlap_ratio_l3655_365569

theorem square_rectangle_overlap_ratio : 
  ∀ (s x y : ℝ),
  s > 0 → x > 0 → y > 0 →
  (0.25 * s^2 = 0.4 * x * y) →
  (y = s) →
  (x / y = 5 / 8) := by
sorry

end square_rectangle_overlap_ratio_l3655_365569


namespace tree_growth_theorem_l3655_365547

/-- Represents the number of branches in Professor Fernando's tree after n weeks -/
def tree_branches : ℕ → ℕ
  | 0 => 0  -- No branches before the tree starts growing
  | 1 => 1  -- One branch in the first week
  | 2 => 1  -- Still one branch in the second week
  | n + 3 => tree_branches (n + 1) + tree_branches (n + 2)  -- Fibonacci recurrence for subsequent weeks

theorem tree_growth_theorem :
  (tree_branches 6 = 8) ∧ 
  (tree_branches 7 = 13) ∧ 
  (tree_branches 13 = 233) := by
sorry

#eval tree_branches 6  -- Expected: 8
#eval tree_branches 7  -- Expected: 13
#eval tree_branches 13  -- Expected: 233

end tree_growth_theorem_l3655_365547


namespace book_distribution_theorem_l3655_365538

def distribute_books (n : ℕ) : ℕ :=
  if n ≥ 2 then n - 1 else 0

theorem book_distribution_theorem :
  distribute_books 8 = 7 :=
by
  sorry

end book_distribution_theorem_l3655_365538


namespace complex_modulus_l3655_365541

theorem complex_modulus (i : ℂ) (z : ℂ) : 
  i * i = -1 → z = 2 + 3 * i / (1 - i) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_modulus_l3655_365541


namespace area_of_region_l3655_365590

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 5 = 4*y - 6*x + 9

-- Theorem statement
theorem area_of_region :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    (π * radius^2 = 17 * π) :=
sorry

end area_of_region_l3655_365590


namespace integer_equation_existence_l3655_365510

theorem integer_equation_existence :
  (¬ ∃ (m n : ℕ+), m * (m + 2) = n * (n + 1)) ∧
  (¬ ∃ (m n : ℕ+), m * (m + 3) = n * (n + 1)) ∧
  (∀ k : ℕ+, k ≥ 4 → ∃ (m n : ℕ+), m * (m + k) = n * (n + 1)) :=
by sorry

end integer_equation_existence_l3655_365510


namespace range_of_a_sum_of_a_and_b_l3655_365584

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|
def g (a x : ℝ) : ℝ := a - |x - 2|

-- Theorem 1: If f(x) < g(x) has solutions, then a > 4
theorem range_of_a (a : ℝ) : 
  (∃ x, f x < g a x) → a > 4 := by sorry

-- Theorem 2: If the solution set of f(x) < g(x) is (b, 7/2), then a + b = 6
theorem sum_of_a_and_b (a b : ℝ) : 
  (∀ x, f x < g a x ↔ b < x ∧ x < 7/2) → a + b = 6 := by sorry

end range_of_a_sum_of_a_and_b_l3655_365584


namespace mardi_gras_necklaces_l3655_365593

theorem mardi_gras_necklaces 
  (boudreaux_necklaces : ℕ)
  (rhonda_necklaces : ℕ)
  (latch_necklaces : ℕ)
  (h1 : boudreaux_necklaces = 12)
  (h2 : rhonda_necklaces = boudreaux_necklaces / 2)
  (h3 : latch_necklaces = 3 * rhonda_necklaces - 4)
  : latch_necklaces = 14 := by
  sorry

end mardi_gras_necklaces_l3655_365593


namespace modulo_eleven_residue_l3655_365540

theorem modulo_eleven_residue : (305 + 7 * 44 + 9 * 176 + 6 * 18) % 11 = 6 := by
  sorry

end modulo_eleven_residue_l3655_365540


namespace tangent_circles_locus_l3655_365592

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency relation
def isTangent (c1 c2 : Circle) : Prop :=
  let d := Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2)
  d = c1.radius + c2.radius ∨ d = |c1.radius - c2.radius|

-- Define the locus of points
inductive Locus
  | Hyperbola
  | StraightLine

-- Theorem statement
theorem tangent_circles_locus 
  (O₁ O₂ P : Circle) 
  (h_separate : O₁.center ≠ O₂.center) 
  (h_tangent₁ : isTangent O₁ P) 
  (h_tangent₂ : isTangent O₂ P) :
  (∃ l₁ l₂ : Locus, l₁ = Locus.Hyperbola ∧ l₂ = Locus.Hyperbola) ∨
  (∃ l₁ l₂ : Locus, l₁ = Locus.Hyperbola ∧ l₂ = Locus.StraightLine) :=
sorry

end tangent_circles_locus_l3655_365592


namespace transformation_eventually_repeats_l3655_365527

/-- Represents a transformation step on a sequence of natural numbers -/
def transform (s : List ℕ) : List ℕ :=
  s.map (λ x => s.count x)

/-- Represents the sequence of transformations applied to an initial sequence -/
def transformation_sequence (initial : List ℕ) : ℕ → List ℕ
  | 0 => initial
  | n + 1 => transform (transformation_sequence initial n)

/-- The theorem stating that the transformation sequence will eventually repeat -/
theorem transformation_eventually_repeats (initial : List ℕ) :
  ∃ n : ℕ, transformation_sequence initial n = transformation_sequence initial (n + 1) := by
  sorry

end transformation_eventually_repeats_l3655_365527


namespace fifth_term_of_sequence_l3655_365597

theorem fifth_term_of_sequence (a : ℕ → ℤ) :
  (∀ n : ℕ, a n = 4 * n - 3) →
  a 5 = 17 := by
sorry

end fifth_term_of_sequence_l3655_365597


namespace profit_sharing_l3655_365507

/-- The profit sharing problem -/
theorem profit_sharing
  (mary_investment mike_investment : ℚ)
  (equal_share_ratio investment_share_ratio : ℚ)
  (mary_extra : ℚ)
  (h1 : mary_investment = 650)
  (h2 : mike_investment = 350)
  (h3 : equal_share_ratio = 1/3)
  (h4 : investment_share_ratio = 2/3)
  (h5 : mary_extra = 600)
  : ∃ P : ℚ,
    P / 6 + (mary_investment / (mary_investment + mike_investment)) * (2 * P / 3) -
    (P / 6 + (mike_investment / (mary_investment + mike_investment)) * (2 * P / 3)) = mary_extra ∧
    P = 3000 := by
  sorry

end profit_sharing_l3655_365507


namespace probability_cos_pi_x_geq_half_over_interval_probability_equals_one_third_l3655_365552

/-- The probability that cos(πx) ≥ 1/2 for x uniformly distributed in [-1, 1] -/
theorem probability_cos_pi_x_geq_half_over_interval (x : ℝ) : 
  ℝ := by sorry

/-- The probability is equal to 1/3 -/
theorem probability_equals_one_third : 
  probability_cos_pi_x_geq_half_over_interval = 1/3 := by sorry

end probability_cos_pi_x_geq_half_over_interval_probability_equals_one_third_l3655_365552


namespace cone_height_l3655_365546

/-- The height of a cone with volume 8192π cubic inches and a vertical cross-section vertex angle of 45 degrees is equal to the cube root of 24576 inches. -/
theorem cone_height (V : ℝ) (θ : ℝ) (h : V = 8192 * Real.pi) (angle : θ = 45) :
  ∃ (H : ℝ), H = (24576 : ℝ) ^ (1/3) ∧ V = (1/3) * Real.pi * H^3 := by
  sorry


end cone_height_l3655_365546


namespace skin_cost_problem_l3655_365514

/-- Given two skins with a total value of 2250 rubles, sold with a total profit of 40%,
    where the profit from the first skin is 25% and the profit from the second skin is -50%,
    prove that the cost of the first skin is 2700 rubles and the cost of the second skin is 450 rubles. -/
theorem skin_cost_problem (x : ℝ) (h1 : x + (2250 - x) = 2250) 
  (h2 : 1.25 * x + 0.5 * (2250 - x) = 1.4 * 2250) : x = 2700 ∧ 2250 - x = 450 := by
  sorry

#check skin_cost_problem

end skin_cost_problem_l3655_365514


namespace polynomial_simplification_l3655_365570

theorem polynomial_simplification (w : ℝ) : 
  3*w + 4 - 2*w^2 - 5*w - 6 + w^2 + 7*w + 8 - 3*w^2 = 5*w - 4*w^2 + 6 := by
  sorry

end polynomial_simplification_l3655_365570


namespace no_integer_square_root_Q_l3655_365560

/-- The polynomial Q(x) = x^4 + 8x^3 + 18x^2 + 11x + 27 -/
def Q (x : ℤ) : ℤ := x^4 + 8*x^3 + 18*x^2 + 11*x + 27

/-- Theorem stating that there are no integer values of x for which Q(x) is a perfect square -/
theorem no_integer_square_root_Q :
  ∀ x : ℤ, ¬∃ y : ℤ, Q x = y^2 := by
  sorry

end no_integer_square_root_Q_l3655_365560


namespace small_boxes_packed_l3655_365521

/-- Represents the number of feet of tape used for sealing each type of box --/
def seal_tape_large : ℕ := 4
def seal_tape_medium : ℕ := 2
def seal_tape_small : ℕ := 1

/-- Represents the number of feet of tape used for address label on each box --/
def label_tape : ℕ := 1

/-- Represents the number of large boxes packed --/
def num_large : ℕ := 2

/-- Represents the number of medium boxes packed --/
def num_medium : ℕ := 8

/-- Represents the total amount of tape used in feet --/
def total_tape : ℕ := 44

/-- Calculates the number of small boxes packed --/
def num_small : ℕ := 
  (total_tape - 
   (num_large * (seal_tape_large + label_tape) + 
    num_medium * (seal_tape_medium + label_tape))) / 
  (seal_tape_small + label_tape)

theorem small_boxes_packed : num_small = 5 := by
  sorry

end small_boxes_packed_l3655_365521


namespace fence_length_of_area_200_l3655_365578

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  short_side : ℝ
  area : ℝ
  area_eq : area = 2 * short_side * short_side

/-- The total fence length of the special rectangle -/
def fence_length (r : SpecialRectangle) : ℝ :=
  2 * r.short_side + 2 * r.short_side

/-- Theorem: The fence length of a special rectangle with area 200 is 40 -/
theorem fence_length_of_area_200 :
  ∃ r : SpecialRectangle, r.area = 200 ∧ fence_length r = 40 := by
  sorry


end fence_length_of_area_200_l3655_365578


namespace exists_n_for_root_1000_l3655_365586

theorem exists_n_for_root_1000 : ∃ n : ℕ, (1000 : ℝ) ^ (1 / n) < 1.001 := by
  sorry

end exists_n_for_root_1000_l3655_365586


namespace seashells_count_l3655_365516

theorem seashells_count (sam_shells mary_shells : ℕ) 
  (h1 : sam_shells = 18) 
  (h2 : mary_shells = 47) : 
  sam_shells + mary_shells = 65 := by
  sorry

end seashells_count_l3655_365516


namespace rectangles_may_not_be_similar_squares_always_similar_equilateral_triangles_always_similar_isosceles_right_triangles_always_similar_l3655_365565

-- Define the shapes
structure Square where
  side : ℝ
  side_positive : side > 0

structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

structure IsoscelesRightTriangle where
  leg : ℝ
  leg_positive : leg > 0

structure Rectangle where
  length : ℝ
  width : ℝ
  length_positive : length > 0
  width_positive : width > 0

-- Define similarity
def similar {α : Type*} (x y : α) : Prop := sorry

-- Theorem stating that rectangles may not always be similar
theorem rectangles_may_not_be_similar :
  ∃ (r1 r2 : Rectangle), ¬ similar r1 r2 :=
sorry

-- Theorems stating that other shapes are always similar
theorem squares_always_similar (s1 s2 : Square) :
  similar s1 s2 :=
sorry

theorem equilateral_triangles_always_similar (t1 t2 : EquilateralTriangle) :
  similar t1 t2 :=
sorry

theorem isosceles_right_triangles_always_similar (t1 t2 : IsoscelesRightTriangle) :
  similar t1 t2 :=
sorry

end rectangles_may_not_be_similar_squares_always_similar_equilateral_triangles_always_similar_isosceles_right_triangles_always_similar_l3655_365565


namespace no_integer_solution_l3655_365585

theorem no_integer_solution :
  ∀ (x y z : ℤ), x ≠ 0 → 2 * x^4 + 2 * x^2 * y^2 + y^4 ≠ z^2 := by
  sorry

end no_integer_solution_l3655_365585


namespace max_value_inequality_l3655_365596

theorem max_value_inequality (m n : ℝ) (hm : m ≠ -3) :
  (∀ x : ℝ, x - 3 * Real.log x + 1 ≥ m * Real.log x + n) →
  (∃ k : ℝ, k = (n - 3) / (m + 3) ∧
    k ≤ -Real.log 2 ∧
    ∀ l : ℝ, l = (n - 3) / (m + 3) → l ≤ k) :=
by sorry

end max_value_inequality_l3655_365596


namespace inhabitable_earth_fraction_l3655_365598

-- Define the fraction of Earth's surface that is land
def land_fraction : ℚ := 1 / 5

-- Define the fraction of land that is inhabitable
def inhabitable_land_fraction : ℚ := 1 / 3

-- Theorem: The fraction of Earth's surface that humans can live on is 1/15
theorem inhabitable_earth_fraction :
  land_fraction * inhabitable_land_fraction = 1 / 15 := by
  sorry

end inhabitable_earth_fraction_l3655_365598


namespace sum_of_first_60_digits_l3655_365583

/-- The decimal representation of 1/9999 -/
def decimal_rep : ℚ := 1 / 9999

/-- The sequence of digits in the decimal representation of 1/9999 -/
def digit_sequence : ℕ → ℕ
  | n => match n % 4 with
         | 0 => 0
         | 1 => 0
         | 2 => 0
         | 3 => 1
         | _ => 0  -- This case is technically unreachable

/-- The sum of the first n digits in the sequence -/
def digit_sum (n : ℕ) : ℕ := (List.range n).map digit_sequence |>.sum

theorem sum_of_first_60_digits :
  digit_sum 60 = 15 :=
sorry

end sum_of_first_60_digits_l3655_365583


namespace calculate_expression_l3655_365539

theorem calculate_expression : (18 / (3 + 9 - 6)) * 4 = 12 := by
  sorry

end calculate_expression_l3655_365539


namespace product_of_binary_and_ternary_l3655_365515

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a ternary number represented as a list of trits to its decimal equivalent -/
def ternary_to_decimal (trits : List ℕ) : ℕ :=
  trits.foldr (fun t acc => 3 * acc + t) 0

/-- The binary representation of 1101₂ -/
def binary_num : List Bool := [true, true, false, true]

/-- The ternary representation of 211₃ -/
def ternary_num : List ℕ := [2, 1, 1]

theorem product_of_binary_and_ternary :
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 286 := by
  sorry

end product_of_binary_and_ternary_l3655_365515


namespace rectangle_side_length_l3655_365517

theorem rectangle_side_length (area : ℚ) (side1 : ℚ) (side2 : ℚ) : 
  area = 1/8 → side1 = 1/2 → area = side1 * side2 → side2 = 1/4 := by
  sorry

end rectangle_side_length_l3655_365517


namespace prep_school_cost_l3655_365580

theorem prep_school_cost (cost_per_semester : ℕ) (semesters_per_year : ℕ) (years : ℕ) : 
  cost_per_semester = 20000 → semesters_per_year = 2 → years = 13 →
  cost_per_semester * semesters_per_year * years = 520000 := by
  sorry

end prep_school_cost_l3655_365580


namespace greatest_four_digit_divisible_by_3_and_6_l3655_365573

theorem greatest_four_digit_divisible_by_3_and_6 : ∃ n : ℕ,
  n = 9996 ∧
  n ≥ 1000 ∧ n < 10000 ∧
  n % 3 = 0 ∧ n % 6 = 0 ∧
  ∀ m : ℕ, m > n → m < 10000 → (m % 3 ≠ 0 ∨ m % 6 ≠ 0) :=
by sorry

end greatest_four_digit_divisible_by_3_and_6_l3655_365573


namespace x_squared_is_quadratic_l3655_365588

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem stating that x² = 0 is a quadratic equation -/
theorem x_squared_is_quadratic : is_quadratic_equation f := by
  sorry

end x_squared_is_quadratic_l3655_365588


namespace root_product_plus_one_l3655_365572

theorem root_product_plus_one (p q r : ℂ) : 
  p^3 - 15*p^2 + 10*p + 24 = 0 →
  q^3 - 15*q^2 + 10*q + 24 = 0 →
  r^3 - 15*r^2 + 10*r + 24 = 0 →
  (1+p)*(1+q)*(1+r) = 2 := by
sorry

end root_product_plus_one_l3655_365572


namespace sam_study_time_l3655_365542

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time Sam spends studying Science in minutes -/
def science_time : ℕ := 60

/-- The time Sam spends studying Math in minutes -/
def math_time : ℕ := 80

/-- The time Sam spends studying Literature in minutes -/
def literature_time : ℕ := 40

/-- The total time Sam spends studying in hours -/
def total_study_time : ℚ :=
  (science_time + math_time + literature_time : ℚ) / minutes_per_hour

theorem sam_study_time :
  total_study_time = 3 := by sorry

end sam_study_time_l3655_365542


namespace tip_percentage_is_twenty_percent_l3655_365564

def lunch_cost : ℝ := 60.50
def total_spent : ℝ := 72.6

theorem tip_percentage_is_twenty_percent :
  (total_spent - lunch_cost) / lunch_cost * 100 = 20 := by
  sorry

end tip_percentage_is_twenty_percent_l3655_365564


namespace sandy_bought_six_fish_l3655_365575

/-- The number of fish Sandy bought -/
def fish_bought (initial : ℕ) (current : ℕ) : ℕ := current - initial

/-- Proof that Sandy bought 6 fish -/
theorem sandy_bought_six_fish :
  let initial_fish : ℕ := 26
  let current_fish : ℕ := 32
  fish_bought initial_fish current_fish = 6 := by
  sorry

end sandy_bought_six_fish_l3655_365575


namespace radical_equation_solution_l3655_365556

theorem radical_equation_solution :
  ∃! x : ℝ, x > 9 ∧ Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3 ∧ x = 18 := by
  sorry

end radical_equation_solution_l3655_365556


namespace product_seven_consecutive_divisible_by_100_l3655_365536

theorem product_seven_consecutive_divisible_by_100 (n : ℕ) : 
  100 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) :=
by sorry

end product_seven_consecutive_divisible_by_100_l3655_365536


namespace smallest_even_triangle_perimeter_l3655_365548

/-- A triangle with consecutive even integer side lengths. -/
structure EvenTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h_even : ∃ k : ℕ, a = 2*k ∧ b = 2*(k+1) ∧ c = 2*(k+2)
  h_triangle : a + b > c ∧ a + c > b ∧ b + c > a

/-- The perimeter of an EvenTriangle. -/
def perimeter (t : EvenTriangle) : ℕ := t.a + t.b + t.c

/-- The smallest possible perimeter of an EvenTriangle is 12. -/
theorem smallest_even_triangle_perimeter :
  ∃ t : EvenTriangle, perimeter t = 12 ∧ ∀ t' : EvenTriangle, perimeter t ≤ perimeter t' :=
sorry

end smallest_even_triangle_perimeter_l3655_365548


namespace binomial_coefficient_equality_l3655_365534

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose n 3 = 7 * Nat.choose n 1) → n = 8 := by
  sorry

end binomial_coefficient_equality_l3655_365534


namespace max_at_two_l3655_365524

/-- The function f(x) defined as x(x-c)² --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) with respect to x --/
def f_derivative (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

theorem max_at_two (c : ℝ) :
  (∀ x : ℝ, f c x ≤ f c 2) ↔ c = 6 := by sorry

end max_at_two_l3655_365524


namespace max_value_of_a_l3655_365579

theorem max_value_of_a : ∃ (a_max : ℝ), a_max = 16175 ∧
  ∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 →
    -2022 ≤ (a + 1) * x^2 - (a + 1) * x + 2022 ∧
    (a + 1) * x^2 - (a + 1) * x + 2022 ≤ 2022) →
  a ≤ a_max := by
  sorry

end max_value_of_a_l3655_365579


namespace banana_orange_equivalence_l3655_365594

/-- Given that 2/3 of 10 bananas are worth as much as 8 oranges,
    prove that 1/2 of 5 bananas are worth as much as 3 oranges. -/
theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
    (2 / 3 : ℚ) * 10 * banana_value = 8 * orange_value →
    (1 / 2 : ℚ) * 5 * banana_value = 3 * orange_value := by
  sorry

end banana_orange_equivalence_l3655_365594


namespace soda_survey_result_l3655_365520

-- Define the total number of people surveyed
def total_surveyed : ℕ := 500

-- Define the central angle of the "Soda" sector in degrees
def soda_angle : ℕ := 198

-- Define the function to calculate the number of people who chose "Soda"
def soda_count : ℕ := (total_surveyed * soda_angle) / 360

-- Theorem statement
theorem soda_survey_result : soda_count = 275 := by
  sorry

end soda_survey_result_l3655_365520


namespace similar_triangles_leg_length_l3655_365543

theorem similar_triangles_leg_length :
  ∀ (y : ℝ),
  (12 : ℝ) / y = 9 / 6 →
  y = 8 :=
by
  sorry

end similar_triangles_leg_length_l3655_365543


namespace vector_norm_sum_l3655_365566

theorem vector_norm_sum (a b c : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ‖a‖ = 2) (h2 : ‖b‖ = 2) (h3 : ‖c‖ = 3) : 
  ‖a - 3 • b‖^2 + ‖b - 3 • c‖^2 + ‖c - 3 • a‖^2 = 170 := by
  sorry

end vector_norm_sum_l3655_365566


namespace no_alpha_exists_for_inequality_l3655_365530

theorem no_alpha_exists_for_inequality :
  ∀ α : ℝ, α > 0 → ∃ x : ℝ, |Real.cos x| + |Real.cos (α * x)| ≤ Real.sin x + Real.sin (α * x) := by
  sorry

end no_alpha_exists_for_inequality_l3655_365530


namespace composite_shape_area_theorem_l3655_365562

/-- The composite shape formed by a hexagon and an octagon attached to an equilateral triangle --/
structure CompositeShape where
  sideLength : ℝ
  hexagonArea : ℝ
  octagonArea : ℝ

/-- Calculate the area of the composite shape --/
def compositeShapeArea (shape : CompositeShape) : ℝ :=
  shape.hexagonArea + shape.octagonArea

/-- The theorem stating the area of the composite shape --/
theorem composite_shape_area_theorem (shape : CompositeShape) 
  (h1 : shape.sideLength = 2)
  (h2 : shape.hexagonArea = 4 * Real.sqrt 3 + 6)
  (h3 : shape.octagonArea = 8 * (1 + Real.sqrt 2) - 12) :
  compositeShapeArea shape = 4 * Real.sqrt 3 + 8 * Real.sqrt 2 + 2 := by
  sorry

end composite_shape_area_theorem_l3655_365562


namespace jake_coffee_drop_probability_l3655_365502

theorem jake_coffee_drop_probability 
  (trip_probability : ℝ) 
  (not_drop_probability : ℝ) 
  (h1 : trip_probability = 0.4)
  (h2 : not_drop_probability = 0.9) :
  1 - not_drop_probability = 0.1 :=
by sorry

end jake_coffee_drop_probability_l3655_365502


namespace hexagon_triangle_area_l3655_365531

/-- The area of an equilateral triangle formed by connecting the second, third, and fifth vertices
    of a regular hexagon with side length 12 cm is 36√3 cm^2. -/
theorem hexagon_triangle_area :
  let hexagon_side : ℝ := 12
  let triangle_side : ℝ := hexagon_side
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side ^ 2
  triangle_area = 36 * Real.sqrt 3 := by sorry

end hexagon_triangle_area_l3655_365531


namespace promotion_difference_l3655_365581

/-- Calculates the total cost of two pairs of shoes using Promotion A -/
def costPromotionA (price : ℝ) : ℝ :=
  price + price * 0.4

/-- Calculates the total cost of two pairs of shoes using Promotion B -/
def costPromotionB (price : ℝ) : ℝ :=
  price + (price - 15)

/-- Proves that the difference between Promotion B and Promotion A is $15 -/
theorem promotion_difference (shoe_price : ℝ) (h : shoe_price = 50) :
  costPromotionB shoe_price - costPromotionA shoe_price = 15 := by
  sorry

#eval costPromotionB 50 - costPromotionA 50

end promotion_difference_l3655_365581


namespace basketball_score_proof_l3655_365574

/-- Given two teams in a basketball game where:
  * The total points scored is 50
  * One team wins by a margin of 28 points
  Prove that the losing team scored 11 points -/
theorem basketball_score_proof (total_points winning_margin : ℕ) 
  (h1 : total_points = 50)
  (h2 : winning_margin = 28) :
  ∃ (winner_score loser_score : ℕ),
    winner_score + loser_score = total_points ∧
    winner_score - loser_score = winning_margin ∧
    loser_score = 11 := by
sorry

end basketball_score_proof_l3655_365574


namespace parabola_vertex_l3655_365511

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 3 * (x - 1)^2 + 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = 3(x-1)^2 + 2 is at the point (1, 2) -/
theorem parabola_vertex :
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧
  parabola (vertex.1) = vertex.2 :=
sorry

end parabola_vertex_l3655_365511


namespace stratified_sampling_most_appropriate_l3655_365549

/-- Represents different sampling methods --/
inductive SamplingMethod
  | Lottery
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a population with two subgroups --/
structure Population where
  total_size : ℕ
  subgroup1_size : ℕ
  subgroup2_size : ℕ
  h_size_sum : subgroup1_size + subgroup2_size = total_size

/-- Represents the goal of the sampling --/
inductive SamplingGoal
  | UnderstandSubgroupDifferences

/-- The most appropriate sampling method given a population and a goal --/
def most_appropriate_sampling_method (pop : Population) (goal : SamplingGoal) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is most appropriate for the given scenario --/
theorem stratified_sampling_most_appropriate 
  (pop : Population) 
  (h_equal_subgroups : pop.subgroup1_size = pop.subgroup2_size) 
  (goal : SamplingGoal) 
  (h_goal : goal = SamplingGoal.UnderstandSubgroupDifferences) :
  most_appropriate_sampling_method pop goal = SamplingMethod.Stratified :=
  sorry

end stratified_sampling_most_appropriate_l3655_365549


namespace line_tangent_to_circle_l3655_365518

/-- A line with slope 1 passing through (0, a) is tangent to the circle x^2 + y^2 = 2 if and only if a = ±2 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∃ (x y : ℝ), y = x + a ∧ x^2 + y^2 = 2 ∧ 
  ∀ (x' y' : ℝ), y' = x' + a → x'^2 + y'^2 ≥ 2) ↔ 
  (a = 2 ∨ a = -2) := by
sorry

end line_tangent_to_circle_l3655_365518


namespace exactly_one_success_probability_l3655_365528

theorem exactly_one_success_probability (p : ℝ) (h1 : p = 1/3) : 
  3 * (1 - p) * p^2 = 2/9 := by
  sorry

end exactly_one_success_probability_l3655_365528


namespace parallelogram_diagonal_intersection_l3655_365525

/-- Given a parallelogram with opposite vertices (2, -3) and (14, 9),
    the intersection point of its diagonals is (8, 3). -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (14, 9)
  let midpoint := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 3) := by sorry

end parallelogram_diagonal_intersection_l3655_365525


namespace final_student_count_l3655_365553

theorem final_student_count (initial_students : ℕ) (students_left : ℕ) (new_students : ℕ)
  (h1 : initial_students = 33)
  (h2 : students_left = 18)
  (h3 : new_students = 14) :
  initial_students - students_left + new_students = 29 := by
  sorry

end final_student_count_l3655_365553


namespace consecutive_integers_sum_of_cubes_l3655_365568

theorem consecutive_integers_sum_of_cubes (n : ℕ) : 
  (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 8830 →
  (n - 1)^3 + n^3 + (n + 1)^3 + (n + 2)^3 = 52264 :=
by sorry

end consecutive_integers_sum_of_cubes_l3655_365568


namespace fifth_power_sum_l3655_365544

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a*x + b*y = 5)
  (h2 : a*x^2 + b*y^2 = 9)
  (h3 : a*x^3 + b*y^3 = 22)
  (h4 : a*x^4 + b*y^4 = 60) :
  a*x^5 + b*y^5 = 97089/203 := by
sorry

end fifth_power_sum_l3655_365544


namespace team_formation_proof_l3655_365532

def number_of_teams (total_girls : ℕ) (total_boys : ℕ) (team_girls : ℕ) (team_boys : ℕ) (mandatory_girl : ℕ) : ℕ :=
  Nat.choose (total_girls - mandatory_girl) (team_girls - mandatory_girl) * Nat.choose total_boys team_boys

theorem team_formation_proof :
  let total_girls : ℕ := 5
  let total_boys : ℕ := 7
  let team_girls : ℕ := 2
  let team_boys : ℕ := 2
  let mandatory_girl : ℕ := 1
  number_of_teams total_girls total_boys team_girls team_boys mandatory_girl = 84 :=
by
  sorry

end team_formation_proof_l3655_365532


namespace age_sum_proof_l3655_365509

theorem age_sum_proof (asaf_age : ℕ) (alexander_age : ℕ) 
  (asaf_pencils : ℕ) (alexander_pencils : ℕ) : 
  asaf_age = 50 →
  asaf_age - alexander_age = asaf_pencils / 2 →
  alexander_pencils = asaf_pencils + 60 →
  asaf_pencils + alexander_pencils = 220 →
  asaf_age + alexander_age = 60 := by
sorry

end age_sum_proof_l3655_365509


namespace largest_angle_convex_pentagon_l3655_365529

theorem largest_angle_convex_pentagon (x : ℚ) :
  (x + 2) + (2*x + 3) + (3*x + 6) + (4*x + 5) + (5*x + 4) = 540 →
  max (x + 2) (max (2*x + 3) (max (3*x + 6) (max (4*x + 5) (5*x + 4)))) = 532 / 3 :=
by sorry

end largest_angle_convex_pentagon_l3655_365529


namespace alternate_multiply_divide_result_l3655_365571

def alternateMultiplyDivide (n : ℕ) (initial : ℕ) : ℚ :=
  match n with
  | 0 => initial
  | m + 1 => if m % 2 = 0
             then (alternateMultiplyDivide m initial) * 3
             else (alternateMultiplyDivide m initial) / 2

theorem alternate_multiply_divide_result :
  alternateMultiplyDivide 15 (9^6) = 3^20 / 2^7 := by
  sorry

end alternate_multiply_divide_result_l3655_365571


namespace ball_probability_l3655_365567

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 30)
  (h_yellow : yellow = 10)
  (h_red : red = 7)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 9 / 10 := by
sorry

end ball_probability_l3655_365567


namespace not_A_inter_B_eq_open_closed_interval_l3655_365512

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | |x - 1| > 2}

def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

theorem not_A_inter_B_eq_open_closed_interval : 
  (Aᶜ ∩ B) = {x | 2 < x ∧ x ≤ 3} :=
sorry

end not_A_inter_B_eq_open_closed_interval_l3655_365512


namespace ninth_grade_students_l3655_365551

theorem ninth_grade_students (S : ℕ) : 
  (S / 4 : ℚ) + (3 * S / 4 / 3 : ℚ) + 20 + 70 = S → S = 180 := by
  sorry

end ninth_grade_students_l3655_365551


namespace mistaken_multiplication_l3655_365599

theorem mistaken_multiplication (correct_multiplier : ℕ) (actual_number : ℕ) (difference : ℕ) :
  correct_multiplier = 43 →
  actual_number = 135 →
  actual_number * correct_multiplier - actual_number * (correct_multiplier - (difference / actual_number)) = difference →
  correct_multiplier - (difference / actual_number) = 34 :=
by sorry

end mistaken_multiplication_l3655_365599


namespace condition_nature_l3655_365595

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | |2*x - a| < 2}

-- Theorem statement
theorem condition_nature (a : ℝ) :
  (∀ a, 1 ∈ M a → 0 ≤ a ∧ a ≤ 4) ∧
  (∃ a, 0 ≤ a ∧ a ≤ 4 ∧ 1 ∉ M a) := by
  sorry

end condition_nature_l3655_365595


namespace ellipse_eccentricity_range_l3655_365501

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    left vertex A, top vertex B, right focus F, and midpoint M of AB,
    prove that the eccentricity e is in the range (0, -1+√3] 
    if 2⋅MA⋅MF + |BF|² ≥ 0 -/
theorem ellipse_eccentricity_range (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * (a/2 * (c + a/2) + b/2 * (-b/2)) + (b^2 + c^2) ≥ 0) :
  let e := c / a
  ∃ (e : ℝ), 0 < e ∧ e ≤ -1 + Real.sqrt 3 := by
  sorry

end ellipse_eccentricity_range_l3655_365501


namespace bakery_weekly_sales_l3655_365535

/-- Represents the daily sales of cakes for a specific type -/
structure DailySales :=
  (monday : Nat)
  (tuesday : Nat)
  (wednesday : Nat)
  (thursday : Nat)
  (friday : Nat)
  (saturday : Nat)
  (sunday : Nat)

/-- Represents the weekly sales data for all cake types -/
structure WeeklySales :=
  (chocolate : DailySales)
  (vanilla : DailySales)
  (strawberry : DailySales)

def bakery_sales : WeeklySales :=
  { chocolate := { monday := 6, tuesday := 7, wednesday := 4, thursday := 8, friday := 9, saturday := 10, sunday := 5 },
    vanilla := { monday := 4, tuesday := 5, wednesday := 3, thursday := 7, friday := 6, saturday := 8, sunday := 4 },
    strawberry := { monday := 3, tuesday := 2, wednesday := 6, thursday := 4, friday := 5, saturday := 7, sunday := 4 } }

def total_sales (sales : DailySales) : Nat :=
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday + sales.friday + sales.saturday + sales.sunday

theorem bakery_weekly_sales :
  total_sales bakery_sales.chocolate = 49 ∧
  total_sales bakery_sales.vanilla = 37 ∧
  total_sales bakery_sales.strawberry = 31 := by
  sorry

end bakery_weekly_sales_l3655_365535


namespace complex_product_ab_l3655_365589

theorem complex_product_ab (a b : ℝ) (i : ℂ) (h : i * i = -1) 
  (h1 : (1 - 2*i)*i = a + b*i) : a * b = 2 := by
  sorry

end complex_product_ab_l3655_365589


namespace complex_equation_solution_l3655_365513

theorem complex_equation_solution (a : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (1 + a * i) * i = -3 + i) : a = 3 := by
  sorry

end complex_equation_solution_l3655_365513


namespace even_function_with_domain_l3655_365563

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem even_function_with_domain (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →
  (∀ x, f a b x ≠ 0 → a - 1 ≤ x ∧ x ≤ 2 * a) →
  (∃ c d : ℝ, ∀ x, -2/3 ≤ x ∧ x ≤ 2/3 → f a b x = 1/3 * x^2 + 1) :=
sorry

end even_function_with_domain_l3655_365563


namespace quadratic_form_sum_l3655_365522

theorem quadratic_form_sum (x : ℝ) : ∃ (a h k : ℝ), 
  (6 * x^2 - 24 * x + 10 = a * (x - h)^2 + k) ∧ (a + h + k = -6) := by
  sorry

end quadratic_form_sum_l3655_365522


namespace parabola_c_value_l3655_365557

/-- A parabola with equation x = ay² + by + c, vertex at (5, 3), and passing through (3, 5) has c = 1/2 -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ y : ℝ, 5 = a * 3^2 + b * 3 + c) →  -- vertex condition
  (∀ y : ℝ, 3 = a * 5^2 + b * 5 + c) →  -- point condition
  c = 1/2 := by sorry

end parabola_c_value_l3655_365557


namespace special_number_between_18_and_57_l3655_365554

theorem special_number_between_18_and_57 :
  ∃! n : ℕ, 18 ≤ n ∧ n ≤ 57 ∧ 
  7 ∣ n ∧ 
  (∀ p : ℕ, Prime p → p ≠ 7 → ¬(p ∣ n)) ∧
  n = 49 ∧
  Real.sqrt n = 7 := by
sorry

end special_number_between_18_and_57_l3655_365554


namespace monkey_count_l3655_365587

/-- Given a group of monkeys that can eat 6 bananas in 6 minutes and 18 bananas in 18 minutes,
    prove that there are 6 monkeys in the group. -/
theorem monkey_count (eating_rate : ℕ → ℕ → ℕ) (monkey_count : ℕ) : 
  (eating_rate 6 6 = 6) →  -- 6 bananas in 6 minutes
  (eating_rate 18 18 = 18) →  -- 18 bananas in 18 minutes
  monkey_count = 6 :=
by sorry

end monkey_count_l3655_365587


namespace twenty_first_term_of_ap_l3655_365561

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem twenty_first_term_of_ap (a₁ d : ℝ) (h₁ : a₁ = 3) (h₂ : d = 5) :
  arithmeticProgressionTerm a₁ d 21 = 103 :=
by sorry

end twenty_first_term_of_ap_l3655_365561


namespace total_books_l3655_365508

theorem total_books (joan_books tom_books : ℕ) 
  (h1 : joan_books = 10) 
  (h2 : tom_books = 38) : 
  joan_books + tom_books = 48 := by
sorry

end total_books_l3655_365508


namespace equation_solution_l3655_365582

theorem equation_solution :
  ∃ y : ℚ, (5 * y - 2) / (6 * y - 6) = 3 / 4 ∧ y = -5 := by
  sorry

end equation_solution_l3655_365582


namespace combined_salaries_l3655_365555

/-- Given 5 individuals with an average salary of 8200 and one individual with a salary of 7000,
    prove that the sum of the other 4 individuals' salaries is 34000 -/
theorem combined_salaries (average_salary : ℕ) (num_individuals : ℕ) (d_salary : ℕ) :
  average_salary = 8200 →
  num_individuals = 5 →
  d_salary = 7000 →
  (average_salary * num_individuals) - d_salary = 34000 := by
  sorry

end combined_salaries_l3655_365555


namespace largest_number_proof_l3655_365503

theorem largest_number_proof (a b c : ℝ) 
  (sum_eq : a + b + c = 100)
  (larger_diff : c - b = 8)
  (smaller_diff : b - a = 5) :
  c = 121 / 3 := by
  sorry

end largest_number_proof_l3655_365503


namespace train_length_l3655_365550

/-- Given a train traveling at 45 km/hr, crossing a bridge of 240.03 meters in 30 seconds,
    the length of the train is 134.97 meters. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 →
  bridge_length = 240.03 →
  crossing_time = 30 →
  (train_speed * 1000 / 3600 * crossing_time) - bridge_length = 134.97 := by
  sorry

#eval (45 * 1000 / 3600 * 30) - 240.03

end train_length_l3655_365550


namespace quadratic_inequality_solution_l3655_365591

theorem quadratic_inequality_solution (a : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 → 
  (∀ x, x^2 - 2*a*x - 8*a^2 < 0 ↔ x₁ < x ∧ x < x₂) → 
  x₂ + x₁ = 15 → 
  a = 15/2 := by
sorry

end quadratic_inequality_solution_l3655_365591


namespace partial_fraction_decomposition_l3655_365523

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 →
  (-2 * x^2 + 5 * x - 6) / (x^3 + x) = -6 / x + (4 * x + 5) / (x^2 + 1) :=
by sorry

end partial_fraction_decomposition_l3655_365523


namespace weight_per_hour_is_correct_l3655_365558

/-- Represents the types of coins Jim finds --/
inductive CoinType
| Gold
| Silver
| Bronze

/-- Represents a bag of coins --/
structure CoinBag where
  coinType : CoinType
  count : ℕ

def hours_spent : ℕ := 8

def coin_weight (ct : CoinType) : ℕ :=
  match ct with
  | CoinType.Gold => 10
  | CoinType.Silver => 5
  | CoinType.Bronze => 2

def treasure_chest : CoinBag := ⟨CoinType.Gold, 100⟩
def smaller_bags : List CoinBag := [⟨CoinType.Gold, 50⟩, ⟨CoinType.Gold, 50⟩]
def other_bags : List CoinBag := [⟨CoinType.Gold, 30⟩, ⟨CoinType.Gold, 20⟩, ⟨CoinType.Gold, 10⟩]
def silver_coins : CoinBag := ⟨CoinType.Silver, 30⟩
def bronze_coins : CoinBag := ⟨CoinType.Bronze, 50⟩

def all_bags : List CoinBag :=
  [treasure_chest] ++ smaller_bags ++ other_bags ++ [silver_coins, bronze_coins]

def total_weight (bags : List CoinBag) : ℕ :=
  bags.foldl (fun acc bag => acc + bag.count * coin_weight bag.coinType) 0

theorem weight_per_hour_is_correct :
  (total_weight all_bags : ℚ) / hours_spent = 356.25 := by sorry

end weight_per_hour_is_correct_l3655_365558
