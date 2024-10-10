import Mathlib

namespace area_between_circle_and_square_l2132_213268

/-- Given a square with side length 2 and a circle with radius √2 sharing the same center,
    the area inside the circle but outside the square is equal to 2π - 4. -/
theorem area_between_circle_and_square :
  let square_side : ℝ := 2
  let circle_radius : ℝ := Real.sqrt 2
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  circle_area - square_area = 2 * π - 4 := by
  sorry

end area_between_circle_and_square_l2132_213268


namespace path_length_2x1x1_block_l2132_213283

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the path traced by a dot on the block -/
def path_length (b : Block) : ℝ := sorry

/-- Theorem stating that the path length for a 2×1×1 block is 4π -/
theorem path_length_2x1x1_block :
  let b : Block := ⟨2, 1, 1⟩
  path_length b = 4 * Real.pi := by sorry

end path_length_2x1x1_block_l2132_213283


namespace smallest_common_divisor_l2132_213264

theorem smallest_common_divisor (n : ℕ) (h1 : n = 627) :
  let m := n + 3
  let k := Nat.minFac (Nat.gcd m (Nat.gcd 4590 105))
  k = 105 := by sorry

end smallest_common_divisor_l2132_213264


namespace fraction_simplification_l2132_213227

/-- Proves that for x = 198719871987, the fraction 198719871987 / (x^2 - (x-1)(x+1)) simplifies to 1987 -/
theorem fraction_simplification (x : ℕ) (h : x = 198719871987) :
  (x : ℚ) / (x^2 - (x-1)*(x+1)) = 1987 := by
  sorry

end fraction_simplification_l2132_213227


namespace folded_paper_cut_ratio_l2132_213247

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem folded_paper_cut_ratio :
  let original_side : ℝ := 6
  let folded_paper := Rectangle.mk original_side (original_side / 2)
  let large_rectangle := folded_paper
  let small_rectangle := Rectangle.mk (original_side / 2) (original_side / 2)
  (perimeter small_rectangle) / (perimeter large_rectangle) = 2 / 3 := by
  sorry

end folded_paper_cut_ratio_l2132_213247


namespace water_amount_depends_on_time_l2132_213211

/-- Represents the water amount in the reservoir -/
def water_amount (t : ℝ) : ℝ := 50 - 2 * t

/-- States that water_amount is a function of time -/
theorem water_amount_depends_on_time :
  ∃ (f : ℝ → ℝ), ∀ t, water_amount t = f t :=
sorry

end water_amount_depends_on_time_l2132_213211


namespace division_remainder_l2132_213290

theorem division_remainder (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 30 = 18 := by
  sorry

end division_remainder_l2132_213290


namespace intersection_of_M_and_complement_of_N_l2132_213230

def M : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

def N : Set ℝ := {x | Real.log 2 ^ (1 - x) < 1}

theorem intersection_of_M_and_complement_of_N :
  M ∩ (Set.univ \ N) = Set.Icc 1 2 \ {2} :=
sorry

end intersection_of_M_and_complement_of_N_l2132_213230


namespace smallest_four_digit_congruence_solution_l2132_213299

theorem smallest_four_digit_congruence_solution :
  let x : ℕ := 1001
  (∀ y : ℕ, 1000 ≤ y ∧ y < x →
    ¬(11 * y ≡ 33 [ZMOD 22] ∧
      3 * y + 10 ≡ 19 [ZMOD 12] ∧
      5 * y - 3 ≡ 2 * y [ZMOD 36] ∧
      y ≡ 3 [ZMOD 4])) ∧
  (11 * x ≡ 33 [ZMOD 22] ∧
   3 * x + 10 ≡ 19 [ZMOD 12] ∧
   5 * x - 3 ≡ 2 * x [ZMOD 36] ∧
   x ≡ 3 [ZMOD 4]) :=
by sorry

end smallest_four_digit_congruence_solution_l2132_213299


namespace correct_factorizations_l2132_213200

theorem correct_factorizations (x y : ℝ) : 
  (x^2 + x*y + y^2 ≠ (x + y)^2) ∧ 
  (-x^2 + 2*x*y - y^2 = -(x - y)^2) ∧ 
  (x^2 + 6*x*y - 9*y^2 ≠ (x - 3*y)^2) ∧ 
  (-x^2 + 1/4 = (1/2 + x)*(1/2 - x)) := by
sorry

end correct_factorizations_l2132_213200


namespace ray_walks_dog_three_times_daily_l2132_213217

/-- The number of times Ray walks his dog each day -/
def walks_per_day (route_length total_distance : ℕ) : ℕ :=
  total_distance / route_length

theorem ray_walks_dog_three_times_daily :
  let route_length : ℕ := 4 + 7 + 11
  let total_distance : ℕ := 66
  walks_per_day route_length total_distance = 3 := by
  sorry

end ray_walks_dog_three_times_daily_l2132_213217


namespace present_age_of_R_l2132_213272

-- Define the present ages of P, Q, and R
variable (Pp Qp Rp : ℝ)

-- Define the conditions
def condition1 : Prop := Pp - 8 = (1/2) * (Qp - 8)
def condition2 : Prop := Qp - 8 = (2/3) * (Rp - 8)
def condition3 : Prop := Qp = 2 * Real.sqrt Rp
def condition4 : Prop := Pp / Qp = 3/5

-- Theorem statement
theorem present_age_of_R 
  (h1 : condition1 Pp Qp)
  (h2 : condition2 Qp Rp)
  (h3 : condition3 Qp Rp)
  (h4 : condition4 Pp Qp) :
  Rp = 400 := by
  sorry

end present_age_of_R_l2132_213272


namespace green_peaches_count_l2132_213274

/-- The number of baskets -/
def num_baskets : ℕ := 7

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 2

/-- The total number of green peaches -/
def total_green_peaches : ℕ := num_baskets * green_peaches_per_basket

theorem green_peaches_count : total_green_peaches = 14 := by
  sorry

end green_peaches_count_l2132_213274


namespace b_range_l2132_213282

theorem b_range (b : ℝ) : (∀ x : ℝ, x^2 + b*x + b > 0) → b ∈ Set.Ioo 0 4 := by
  sorry

end b_range_l2132_213282


namespace sam_seashells_l2132_213294

/-- The number of seashells Sam found on the beach -/
def total_seashells : ℕ := 35

/-- The number of seashells Sam gave to Joan -/
def seashells_given : ℕ := 18

/-- The number of seashells Sam has now -/
def seashells_remaining : ℕ := 17

/-- Theorem stating that the total number of seashells Sam found is equal to
    the sum of seashells given away and seashells remaining -/
theorem sam_seashells : 
  total_seashells = seashells_given + seashells_remaining := by
  sorry

end sam_seashells_l2132_213294


namespace intersection_equals_N_l2132_213221

-- Define the sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem intersection_equals_N : M ∩ N = N := by sorry

end intersection_equals_N_l2132_213221


namespace square_sum_greater_than_quarter_l2132_213256

theorem square_sum_greater_than_quarter (a b : ℝ) (h : a + b = 1) :
  a^2 + b^2 > 1/4 := by
sorry

end square_sum_greater_than_quarter_l2132_213256


namespace thirtieth_term_is_119_l2132_213297

/-- An arithmetic sequence is defined by its first term and common difference -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ := fun n => a₁ + (n - 1) * d

/-- The first term of our sequence -/
def a₁ : ℝ := 3

/-- The second term of our sequence -/
def a₂ : ℝ := 7

/-- The third term of our sequence -/
def a₃ : ℝ := 11

/-- The common difference of our sequence -/
def d : ℝ := a₂ - a₁

/-- The 30th term of our sequence -/
def a₃₀ : ℝ := arithmeticSequence a₁ d 30

theorem thirtieth_term_is_119 : a₃₀ = 119 := by sorry

end thirtieth_term_is_119_l2132_213297


namespace cubic_factorization_l2132_213263

theorem cubic_factorization (x : ℝ) : -3*x + 6*x^2 - 3*x^3 = -3*x*(x-1)^2 := by
  sorry

end cubic_factorization_l2132_213263


namespace rectangular_to_polar_conversion_l2132_213234

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 2 * Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  r = Real.sqrt 76 ∧ θ = Real.pi / 6 := by
  sorry

end rectangular_to_polar_conversion_l2132_213234


namespace polynomial_expansion_l2132_213267

theorem polynomial_expansion :
  ∀ x : ℝ, (4 * x^3 - 3 * x^2 + 2 * x + 7) * (5 * x^4 + x^3 - 3 * x + 9) =
    20 * x^7 - 27 * x^5 + 8 * x^4 + 45 * x^3 - 4 * x^2 + 51 * x + 196 := by
  sorry

end polynomial_expansion_l2132_213267


namespace range_of_a_l2132_213275

-- Define the function f(x) for any real a
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 + a) * x^2 - a * x + 1

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f a x > 0) ↔ ((-4/3 < a ∧ a < -1) ∨ a = 0) := by
  sorry

end range_of_a_l2132_213275


namespace sum_positive_implies_at_least_one_positive_l2132_213261

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) :
  a + b > 0 → a > 0 ∨ b > 0 := by
  sorry

end sum_positive_implies_at_least_one_positive_l2132_213261


namespace race_probability_l2132_213259

theorem race_probability (total_cars : ℕ) (prob_Y prob_Z prob_XYZ : ℚ) : 
  total_cars = 16 →
  prob_Y = 1/12 →
  prob_Z = 1/7 →
  prob_XYZ = 47619047619047616/100000000000000000 →
  ∃ (prob_X : ℚ), 
    prob_X + prob_Y + prob_Z = prob_XYZ ∧
    prob_X = 1/4 :=
by sorry

end race_probability_l2132_213259


namespace fish_remaining_l2132_213238

theorem fish_remaining (guppies angelfish tiger_sharks oscar_fish : ℕ)
  (guppies_sold angelfish_sold tiger_sharks_sold oscar_fish_sold : ℕ)
  (h1 : guppies = 94)
  (h2 : angelfish = 76)
  (h3 : tiger_sharks = 89)
  (h4 : oscar_fish = 58)
  (h5 : guppies_sold = 30)
  (h6 : angelfish_sold = 48)
  (h7 : tiger_sharks_sold = 17)
  (h8 : oscar_fish_sold = 24) :
  (guppies - guppies_sold) + (angelfish - angelfish_sold) +
  (tiger_sharks - tiger_sharks_sold) + (oscar_fish - oscar_fish_sold) = 198 :=
by sorry

end fish_remaining_l2132_213238


namespace intersection_A_complement_B_l2132_213292

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 3}

-- Theorem statement
theorem intersection_A_complement_B : 
  A ∩ (U \ B) = {x : ℝ | -2 ≤ x ∧ x < 3} := by sorry

end intersection_A_complement_B_l2132_213292


namespace cubic_repeated_root_l2132_213239

/-- The cubic equation has a repeated root iff p = 5 or p = -7 -/
theorem cubic_repeated_root (p : ℝ) : 
  (∃ x : ℝ, (3 * x^3 - (p + 1) * x^2 + 4 * x - 12 = 0) ∧ 
   (6 * x^2 - 2 * (p + 1) * x + 4 = 0)) ↔ 
  (p = 5 ∨ p = -7) :=
sorry

end cubic_repeated_root_l2132_213239


namespace intersection_point_of_lines_l2132_213210

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Defines a line in 2D space using the equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : IntersectionPoint) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- The theorem stating that (2, 1) is the unique intersection point of y = 3 - x and y = 3x - 5 -/
theorem intersection_point_of_lines :
  ∃! p : IntersectionPoint, 
    (pointOnLine p ⟨-1, 3⟩) ∧ (pointOnLine p ⟨3, -5⟩) ∧ p.x = 2 ∧ p.y = 1 :=
by
  sorry

end intersection_point_of_lines_l2132_213210


namespace situps_total_l2132_213280

/-- The number of sit-ups Barney can perform in one minute -/
def barney_situps : ℕ := 45

/-- The number of minutes Barney performs sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie performs sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie performs sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The number of sit-ups Carrie can perform in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can perform in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := barney_situps * barney_minutes + 
                        carrie_situps * carrie_minutes + 
                        jerrie_situps * jerrie_minutes

theorem situps_total : total_situps = 510 := by
  sorry

end situps_total_l2132_213280


namespace FMF_better_than_MFM_l2132_213252

/-- Represents the probability of winning a tennis match against a parent. -/
structure ParentProbability where
  /-- The probability of winning against the parent. -/
  prob : ℝ
  /-- The probability is between 0 and 1. -/
  prob_between_zero_and_one : 0 ≤ prob ∧ prob ≤ 1

/-- Calculates the probability of winning in a Father-Mother-Father (FMF) sequence. -/
def prob_win_FMF (p q : ParentProbability) : ℝ :=
  2 * p.prob * q.prob - p.prob * q.prob^2

/-- Calculates the probability of winning in a Mother-Father-Mother (MFM) sequence. -/
def prob_win_MFM (p q : ParentProbability) : ℝ :=
  2 * p.prob * q.prob - p.prob^2 * q.prob

/-- 
Theorem: The probability of winning in the Father-Mother-Father (FMF) sequence
is higher than the probability of winning in the Mother-Father-Mother (MFM) sequence,
given that the probability of winning against the father is less than
the probability of winning against the mother.
-/
theorem FMF_better_than_MFM (p q : ParentProbability) 
  (h : p.prob < q.prob) : prob_win_FMF p q > prob_win_MFM p q := by
  sorry


end FMF_better_than_MFM_l2132_213252


namespace max_regions_formula_l2132_213269

/-- The maximum number of regions delimited by n lines in a plane -/
def max_regions (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem stating the maximum number of regions delimited by n lines in a plane -/
theorem max_regions_formula (n : ℕ) :
  max_regions n = 1 + n * (n + 1) / 2 := by
  sorry

end max_regions_formula_l2132_213269


namespace rice_division_l2132_213213

theorem rice_division (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 29 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight / num_containers) * ounces_per_pound = 29 := by
sorry

end rice_division_l2132_213213


namespace peach_difference_l2132_213288

def steven_peaches : ℕ := 13
def jake_peaches : ℕ := 7

theorem peach_difference : steven_peaches - jake_peaches = 6 := by
  sorry

end peach_difference_l2132_213288


namespace student_average_age_l2132_213235

theorem student_average_age 
  (num_students : ℕ) 
  (teacher_age : ℕ) 
  (avg_increase : ℝ) : 
  num_students = 22 → 
  teacher_age = 44 → 
  avg_increase = 1 → 
  (((num_students : ℝ) * x + teacher_age) / (num_students + 1) = x + avg_increase) → 
  x = 21 :=
by sorry

end student_average_age_l2132_213235


namespace parabola_intersects_line_segment_l2132_213287

/-- Parabola C_m: y = x^2 - mx + m + 1 -/
def C_m (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m + 1

/-- Line segment AB with endpoints A(0,4) and B(4,0) -/
def line_AB (x : ℝ) : ℝ := -x + 4

/-- The parabola C_m intersects the line segment AB at exactly two points
    if and only if m is in the range [3, 17/3] -/
theorem parabola_intersects_line_segment (m : ℝ) :
  (∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 4 ∧
   C_m m x₁ = line_AB x₁ ∧ C_m m x₂ = line_AB x₂ ∧
   ∀ x, 0 ≤ x ∧ x ≤ 4 → C_m m x = line_AB x → (x = x₁ ∨ x = x₂)) ↔
  (3 ≤ m ∧ m ≤ 17/3) :=
sorry

end parabola_intersects_line_segment_l2132_213287


namespace satisfying_polynomial_iff_quadratic_l2132_213223

/-- A polynomial that satisfies the given functional equation -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, P (a + b - 2*c) + P (b + c - 2*a) + P (a + c - 2*b) = 
               3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

/-- The theorem stating the equivalence between the functional equation and the quadratic form -/
theorem satisfying_polynomial_iff_quadratic :
  ∀ P : ℝ → ℝ, SatisfyingPolynomial P ↔ 
    ∃ a b : ℝ, ∀ x : ℝ, P x = a * x^2 + b * x :=
by sorry

end satisfying_polynomial_iff_quadratic_l2132_213223


namespace new_numbers_average_l2132_213228

theorem new_numbers_average (initial_count : ℕ) (initial_mean : ℝ) 
  (new_count : ℕ) (new_mean : ℝ) : 
  initial_count = 12 →
  initial_mean = 45 →
  new_count = 15 →
  new_mean = 60 →
  (new_count * new_mean - initial_count * initial_mean) / (new_count - initial_count) = 120 :=
by sorry

end new_numbers_average_l2132_213228


namespace correct_system_l2132_213224

/-- Represents the price of a horse in taels -/
def horse_price : ℝ := sorry

/-- Represents the price of a head of cattle in taels -/
def cattle_price : ℝ := sorry

/-- The total price of 4 horses and 6 heads of cattle is 48 taels -/
axiom eq1 : 4 * horse_price + 6 * cattle_price = 48

/-- The total price of 3 horses and 5 heads of cattle is 38 taels -/
axiom eq2 : 3 * horse_price + 5 * cattle_price = 38

/-- The system of equations correctly represents the given conditions -/
theorem correct_system : 
  (4 * horse_price + 6 * cattle_price = 48) ∧ 
  (3 * horse_price + 5 * cattle_price = 38) :=
sorry

end correct_system_l2132_213224


namespace matrix_power_eigen_l2132_213243

theorem matrix_power_eigen (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B.vecMul (![3, -1]) = ![12, -4] →
  (B^4).vecMul (![3, -1]) = ![768, -256] := by
  sorry

end matrix_power_eigen_l2132_213243


namespace foreign_stamps_count_l2132_213216

/-- Represents a stamp collection with various properties -/
structure StampCollection where
  total : ℕ
  old : ℕ
  foreignAndOld : ℕ
  neitherForeignNorOld : ℕ

/-- Calculates the number of foreign stamps in the collection -/
def foreignStamps (sc : StampCollection) : ℕ :=
  sc.total - sc.neitherForeignNorOld - (sc.old - sc.foreignAndOld)

/-- Theorem stating the number of foreign stamps in the given collection -/
theorem foreign_stamps_count (sc : StampCollection) 
    (h1 : sc.total = 200)
    (h2 : sc.old = 70)
    (h3 : sc.foreignAndOld = 20)
    (h4 : sc.neitherForeignNorOld = 60) :
    foreignStamps sc = 90 := by
  sorry

#eval foreignStamps { total := 200, old := 70, foreignAndOld := 20, neitherForeignNorOld := 60 }

end foreign_stamps_count_l2132_213216


namespace x_y_inequality_l2132_213231

theorem x_y_inequality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + 2 * |y| = 2 * x * y) : 
  (x > 0 → x + y > 3) ∧ (x < 0 → x + y < -3) := by
  sorry

end x_y_inequality_l2132_213231


namespace arithmetic_sequence_common_difference_l2132_213289

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) 
  (h_seq : ArithmeticSequence a) 
  (h_prod : a 7 * a 11 = 6) 
  (h_sum : a 4 + a 14 = 5) : 
  ∃ d : ℚ, (d = 1/4 ∨ d = -1/4) ∧ ∀ n : ℕ, a (n + 1) = a n + d := by
sorry

end arithmetic_sequence_common_difference_l2132_213289


namespace largest_divisor_of_consecutive_even_products_l2132_213206

theorem largest_divisor_of_consecutive_even_products (n : ℕ+) : 
  let Q := (2 * n) * (2 * n + 2) * (2 * n + 4)
  ∃ k : ℕ, Q = 12 * k ∧ ∀ m : ℕ, m > 12 → ¬(∀ n : ℕ+, m ∣ ((2 * n) * (2 * n + 2) * (2 * n + 4))) :=
by sorry

end largest_divisor_of_consecutive_even_products_l2132_213206


namespace area_triangle_BCD_l2132_213296

/-- Given a triangle ABC with area 50 square units and base AC of 6 units,
    and an extension of AC to point D such that CD is 36 units long,
    prove that the area of triangle BCD is 300 square units. -/
theorem area_triangle_BCD (h : ℝ) : 
  (1/2 : ℝ) * 6 * h = 50 →  -- Area of triangle ABC
  (1/2 : ℝ) * 36 * h = 300  -- Area of triangle BCD
  := by sorry

end area_triangle_BCD_l2132_213296


namespace y_days_to_finish_work_l2132_213249

/-- The number of days x needs to finish the work alone -/
def x_days : ℝ := 18

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 10

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℝ := 6

/-- The number of days y needs to finish the work alone -/
def y_days : ℝ := 15

/-- Theorem stating that given the conditions, y needs 15 days to finish the work alone -/
theorem y_days_to_finish_work : 
  (1 / x_days) * x_remaining = 1 - (y_worked / y_days) := by sorry

end y_days_to_finish_work_l2132_213249


namespace binomial_coeff_not_coprime_l2132_213240

theorem binomial_coeff_not_coprime (n m k : ℕ) (h1 : 0 < k) (h2 : k < m) (h3 : m < n) :
  ∃ d : ℕ, d > 1 ∧ d ∣ Nat.choose n k ∧ d ∣ Nat.choose n m :=
by sorry

end binomial_coeff_not_coprime_l2132_213240


namespace coin_division_problem_l2132_213226

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) →
  (n % 8 = 6) →
  (n % 7 = 5) →
  (n % 9 = 0) := by
sorry

end coin_division_problem_l2132_213226


namespace solution_l2132_213233

def problem (f : ℝ → ℝ) : Prop :=
  (∀ x, f x * f (x + 2) = 13) ∧ f 1 = 2

theorem solution (f : ℝ → ℝ) (h : problem f) : f 2015 = 13/2 := by
  sorry

end solution_l2132_213233


namespace cricketer_matches_count_l2132_213295

/-- Proves that a cricketer played 10 matches given the average scores for all matches, 
    the first 6 matches, and the last 4 matches. -/
theorem cricketer_matches_count 
  (total_average : ℝ) 
  (first_six_average : ℝ) 
  (last_four_average : ℝ) 
  (h1 : total_average = 38.9)
  (h2 : first_six_average = 42)
  (h3 : last_four_average = 34.25) : 
  ∃ (n : ℕ), n = 10 ∧ 
    n * total_average = 6 * first_six_average + 4 * last_four_average := by
  sorry

#check cricketer_matches_count

end cricketer_matches_count_l2132_213295


namespace bell_peppers_needed_l2132_213248

/-- Represents the number of slices and pieces obtained from one bell pepper -/
def slices_per_pepper : ℕ := 20

/-- Represents the fraction of large slices that are cut into smaller pieces -/
def fraction_cut : ℚ := 1/2

/-- Represents the number of smaller pieces each large slice is cut into -/
def pieces_per_slice : ℕ := 3

/-- Represents the total number of slices and pieces Tamia wants to use -/
def total_slices : ℕ := 200

/-- Proves that 5 bell peppers are needed to produce 200 slices and pieces -/
theorem bell_peppers_needed : 
  (total_slices : ℚ) / ((1 - fraction_cut) * slices_per_pepper + 
  fraction_cut * slices_per_pepper * pieces_per_slice) = 5 := by
sorry

end bell_peppers_needed_l2132_213248


namespace range_of_a_in_p_a_neither_necessary_nor_sufficient_for_b_l2132_213270

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define sets A and B
def A : Set ℝ := {a | a ≤ 1}
def B : Set ℝ := {a | a ≥ 1 ∨ a ≤ -2}

-- Theorem for the range of a in proposition p
theorem range_of_a_in_p : ∀ a : ℝ, p a ↔ a ∈ A := by sorry

-- Theorem for the relationship between A and B
theorem a_neither_necessary_nor_sufficient_for_b :
  (¬∀ a : ℝ, a ∈ B → a ∈ A) ∧ (¬∀ a : ℝ, a ∈ A → a ∈ B) := by sorry

end range_of_a_in_p_a_neither_necessary_nor_sufficient_for_b_l2132_213270


namespace max_integer_difference_l2132_213232

theorem max_integer_difference (x y : ℤ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) : 
  (∀ a b : ℤ, 7 < a ∧ a < 9 ∧ 9 < b ∧ b < 15 → y - x ≥ b - a) ∧ y - x = 6 := by
  sorry

end max_integer_difference_l2132_213232


namespace average_weight_a_b_l2132_213253

/-- Given three weights a, b, and c, proves that the average of a and b is 40,
    under certain conditions. -/
theorem average_weight_a_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 43 →
  b = 31 →
  (a + b) / 2 = 40 := by
sorry

end average_weight_a_b_l2132_213253


namespace frank_bought_five_chocolates_l2132_213262

-- Define the cost of items
def chocolate_cost : ℕ := 2
def chips_cost : ℕ := 3

-- Define the number of bags of chips
def chips_count : ℕ := 2

-- Define the total amount spent
def total_spent : ℕ := 16

-- Define the function to calculate the number of chocolate bars
def chocolate_bars : ℕ → Prop
  | n => chocolate_cost * n + chips_cost * chips_count = total_spent

-- Theorem statement
theorem frank_bought_five_chocolates : 
  ∃ (n : ℕ), chocolate_bars n ∧ n = 5 := by sorry

end frank_bought_five_chocolates_l2132_213262


namespace car_collision_frequency_l2132_213284

theorem car_collision_frequency :
  ∀ (x : ℝ),
    (x > 0) →
    (240 / x + 240 / 20 = 36) →
    x = 10 :=
by
  sorry

#check car_collision_frequency

end car_collision_frequency_l2132_213284


namespace least_four_digit_multiple_l2132_213258

theorem least_four_digit_multiple : ∀ n : ℕ,
  (1000 ≤ n) →
  (n % 3 = 0) →
  (n % 4 = 0) →
  (n % 9 = 0) →
  1008 ≤ n :=
by sorry

end least_four_digit_multiple_l2132_213258


namespace system_solution_unique_l2132_213278

theorem system_solution_unique (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 / x + 2 / y = 4 ∧ 5 / x - 6 / y = 2) ↔ (x = 1 ∧ y = 2) :=
by sorry

end system_solution_unique_l2132_213278


namespace composition_value_l2132_213273

-- Define the functions h and j
def h (x : ℝ) : ℝ := 4 * x + 5
def j (x : ℝ) : ℝ := 6 * x - 11

-- State the theorem
theorem composition_value : j (h 5) = 139 := by sorry

end composition_value_l2132_213273


namespace movie_theater_revenue_l2132_213281

/-- Calculates the total revenue of a movie theater given ticket sales and pricing information. -/
def calculate_total_revenue (
  matinee_price : ℚ)
  (evening_price : ℚ)
  (threeD_price : ℚ)
  (evening_group_discount : ℚ)
  (threeD_online_surcharge : ℚ)
  (early_bird_discount : ℚ)
  (matinee_tickets : ℕ)
  (early_bird_tickets : ℕ)
  (evening_tickets : ℕ)
  (evening_group_tickets : ℕ)
  (threeD_tickets : ℕ)
  (threeD_online_tickets : ℕ) : ℚ :=
  sorry

theorem movie_theater_revenue :
  let matinee_price : ℚ := 5
  let evening_price : ℚ := 12
  let threeD_price : ℚ := 20
  let evening_group_discount : ℚ := 0.1
  let threeD_online_surcharge : ℚ := 2
  let early_bird_discount : ℚ := 0.5
  let matinee_tickets : ℕ := 200
  let early_bird_tickets : ℕ := 20
  let evening_tickets : ℕ := 300
  let evening_group_tickets : ℕ := 150
  let threeD_tickets : ℕ := 100
  let threeD_online_tickets : ℕ := 60
  calculate_total_revenue
    matinee_price evening_price threeD_price
    evening_group_discount threeD_online_surcharge early_bird_discount
    matinee_tickets early_bird_tickets evening_tickets
    evening_group_tickets threeD_tickets threeD_online_tickets = 6490 := by
  sorry

end movie_theater_revenue_l2132_213281


namespace biquadratic_root_negation_l2132_213260

/-- A biquadratic equation is of the form ax^4 + bx^2 + c = 0 -/
def BiquadraticEquation (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^4 + b * x^2 + c = 0

/-- If α is a root of a biquadratic equation, then -α is also a root -/
theorem biquadratic_root_negation (a b c α : ℝ) :
  BiquadraticEquation a b c α → BiquadraticEquation a b c (-α) :=
by sorry

end biquadratic_root_negation_l2132_213260


namespace geometric_sequence_sum_l2132_213241

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  property1 : a 3 * a 7 = 8
  property2 : a 4 + a 6 = 6

/-- Theorem: For a geometric sequence satisfying the given properties, a_2 + a_8 = 9 -/
theorem geometric_sequence_sum (seq : GeometricSequence) : seq.a 2 + seq.a 8 = 9 := by
  sorry

end geometric_sequence_sum_l2132_213241


namespace square_9801_property_l2132_213203

theorem square_9801_property (y : ℤ) (h : y^2 = 9801) : (y + 2) * (y - 2) = 9797 := by
  sorry

end square_9801_property_l2132_213203


namespace max_value_abc_l2132_213246

theorem max_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_3 : a + b + c = 3) :
  a^2 * b^3 * c ≤ 27/16 := by
sorry

end max_value_abc_l2132_213246


namespace probability_green_in_specific_bag_l2132_213204

structure Bag where
  total_balls : ℕ
  green_balls : ℕ
  white_balls : ℕ

def probability_green (b : Bag) : ℚ :=
  b.green_balls / b.total_balls

theorem probability_green_in_specific_bag : 
  ∃ (b : Bag), b.total_balls = 9 ∧ b.green_balls = 7 ∧ b.white_balls = 2 ∧ 
    probability_green b = 7 / 9 := by
  sorry

end probability_green_in_specific_bag_l2132_213204


namespace unique_nine_digit_number_l2132_213285

def is_nine_digit (n : ℕ) : Prop := 100000000 ≤ n ∧ n ≤ 999999999

def sum_of_digits (n : ℕ) : ℕ := sorry

def product_of_digits (n : ℕ) : ℕ := sorry

def round_to_millions (n : ℕ) : ℕ := sorry

theorem unique_nine_digit_number :
  ∃! n : ℕ,
    is_nine_digit n ∧
    n % 2 = 1 ∧
    sum_of_digits n = 10 ∧
    product_of_digits n ≠ 0 ∧
    n % 7 = 0 ∧
    round_to_millions n = 112 :=
by sorry

end unique_nine_digit_number_l2132_213285


namespace inequality_solution_set_l2132_213244

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 7| ≥ a^2 - 3*a) → 
  a ∈ Set.Icc (-2 : ℝ) 5 :=
by sorry

end inequality_solution_set_l2132_213244


namespace odd_number_1991_in_group_32_l2132_213220

/-- The n-th group of odd numbers contains (2n-1) numbers -/
def group_size (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of odd numbers up to the n-th group -/
def sum_up_to_group (n : ℕ) : ℕ := n^2

/-- The position of 1991 in the sequence of odd numbers -/
def target : ℕ := 1991

/-- The theorem stating that 1991 is in the 32nd group -/
theorem odd_number_1991_in_group_32 :
  ∃ (n : ℕ), n = 32 ∧ 
  sum_up_to_group (n - 1) < target ∧ 
  target ≤ sum_up_to_group n :=
sorry

end odd_number_1991_in_group_32_l2132_213220


namespace larger_sphere_radius_l2132_213265

/-- The radius of a sphere with volume equal to 12 spheres of radius 0.5 inches is ³√3 inches. -/
theorem larger_sphere_radius (r : ℝ) : 
  (4 / 3 * Real.pi * r^3 = 12 * (4 / 3 * Real.pi * (1 / 2)^3)) → r = (3 : ℝ)^(1/3) :=
by sorry

end larger_sphere_radius_l2132_213265


namespace lcm_36_90_l2132_213286

theorem lcm_36_90 : Nat.lcm 36 90 = 180 := by
  sorry

end lcm_36_90_l2132_213286


namespace a_closed_form_l2132_213298

def a : ℕ → ℤ
  | 0 => -1
  | 1 => 1
  | (n + 2) => 2 * a (n + 1) + 3 * a n + 3^(n + 2)

theorem a_closed_form (n : ℕ) :
  a n = (1 / 16) * ((4 * n - 3) * 3^(n + 1) - 7 * (-1)^n) :=
by sorry

end a_closed_form_l2132_213298


namespace equal_coins_after_transfer_l2132_213279

/-- Represents the amount of gold coins each merchant has -/
structure Merchants where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (m : Merchants) : Prop :=
  (m.ierema + 70 = m.yuliy) ∧ (m.foma - 40 = m.yuliy)

/-- The theorem to prove -/
theorem equal_coins_after_transfer (m : Merchants) 
  (h : satisfies_conditions m) : 
  m.foma - 55 = m.ierema + 55 := by
  sorry

#check equal_coins_after_transfer

end equal_coins_after_transfer_l2132_213279


namespace eccentricity_conic_sections_l2132_213271

theorem eccentricity_conic_sections : ∃ (e₁ e₂ : ℝ), 
  e₁^2 - 5*e₁ + 1 = 0 ∧ 
  e₂^2 - 5*e₂ + 1 = 0 ∧ 
  (0 < e₁ ∧ e₁ < 1) ∧ 
  (e₂ > 1) := by sorry

end eccentricity_conic_sections_l2132_213271


namespace distribute_four_men_five_women_l2132_213205

/-- The number of ways to distribute men and women into groups -/
def distribute_people (num_men num_women : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating the correct number of distributions for 4 men and 5 women -/
theorem distribute_four_men_five_women :
  distribute_people 4 5 = 560 := by
  sorry

end distribute_four_men_five_women_l2132_213205


namespace complex_equation_solution_l2132_213250

theorem complex_equation_solution (z : ℂ) :
  (2 - 3*I)*z = 5 - I → z = 1 + I := by
  sorry

end complex_equation_solution_l2132_213250


namespace tangent_line_sum_l2132_213209

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0

-- State the theorem
theorem tangent_line_sum (h : tangent_line 1 (f 1)) : f 1 + deriv f 1 = 5/3 := by
  sorry

end tangent_line_sum_l2132_213209


namespace angle_with_same_terminal_side_as_negative_415_l2132_213201

-- Define the set of angles with the same terminal side as -415°
def same_terminal_side (β : ℝ) : Prop :=
  ∃ k : ℤ, β = k * 360 - 415

-- Define the condition for the angle to be between 0° and 360°
def between_0_and_360 (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 360

-- Theorem statement
theorem angle_with_same_terminal_side_as_negative_415 :
  ∃ θ : ℝ, same_terminal_side θ ∧ between_0_and_360 θ ∧ θ = 305 :=
by sorry

end angle_with_same_terminal_side_as_negative_415_l2132_213201


namespace max_sum_of_first_two_l2132_213236

theorem max_sum_of_first_two (a b c d e : ℕ) : 
  a < b → b < c → c < d → d < e → 
  a + 2*b + 3*c + 4*d + 5*e = 300 → 
  a + b ≤ 35 :=
by sorry

end max_sum_of_first_two_l2132_213236


namespace sample_size_calculation_l2132_213207

/-- Represents the staff composition in a company -/
structure StaffComposition where
  sales : ℕ
  management : ℕ
  logistics : ℕ

/-- Represents a stratified sample from the company -/
structure StratifiedSample where
  total_size : ℕ
  sales_size : ℕ

/-- The theorem stating the relationship between the company's staff composition,
    the number of sales staff in the sample, and the total sample size -/
theorem sample_size_calculation 
  (company : StaffComposition)
  (sample : StratifiedSample)
  (h1 : company.sales = 15)
  (h2 : company.management = 3)
  (h3 : company.logistics = 2)
  (h4 : sample.sales_size = 30) :
  sample.total_size = 40 := by
  sorry

end sample_size_calculation_l2132_213207


namespace roof_dimension_difference_l2132_213218

theorem roof_dimension_difference :
  ∀ (width length : ℝ),
  width > 0 →
  length = 4 * width →
  width * length = 768 →
  length - width = 24 * Real.sqrt 3 :=
by
  sorry

end roof_dimension_difference_l2132_213218


namespace max_female_students_with_four_teachers_min_group_size_exists_min_group_l2132_213291

/-- Represents the composition of a study group --/
structure StudyGroup where
  male_students : ℕ
  female_students : ℕ
  teachers : ℕ

/-- Checks if a study group satisfies the given conditions --/
def is_valid_group (g : StudyGroup) : Prop :=
  g.male_students > g.female_students ∧
  g.female_students > g.teachers ∧
  2 * g.teachers > g.male_students

theorem max_female_students_with_four_teachers :
  ∀ g : StudyGroup,
  is_valid_group g → g.teachers = 4 →
  g.female_students ≤ 6 :=
sorry

theorem min_group_size :
  ∀ g : StudyGroup,
  is_valid_group g →
  g.male_students + g.female_students + g.teachers ≥ 12 :=
sorry

theorem exists_min_group :
  ∃ g : StudyGroup,
  is_valid_group g ∧
  g.male_students + g.female_students + g.teachers = 12 :=
sorry

end max_female_students_with_four_teachers_min_group_size_exists_min_group_l2132_213291


namespace cylinder_volume_relation_l2132_213215

/-- Proves that the volume of cylinder C is (1/9) π h³ given the specified conditions --/
theorem cylinder_volume_relation (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  h = 3 * r →  -- Height of C is three times radius of D
  r = h →      -- Radius of D is equal to height of C
  π * r^2 * h = 3 * (π * h^2 * r) →  -- Volume of C is three times volume of D
  π * r^2 * h = (1/9) * π * h^3 :=
by
  sorry

end cylinder_volume_relation_l2132_213215


namespace draw_one_is_random_event_l2132_213293

/-- A set of cards numbered from 1 to 10 -/
def CardSet : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}

/-- Definition of a random event -/
def IsRandomEvent (event : Set ℕ → Prop) : Prop :=
  ∃ (s : Set ℕ), event s ∧ ∃ (t : Set ℕ), ¬event t

/-- Drawing a card numbered 1 from the set -/
def DrawOne (s : Set ℕ) : Prop := 1 ∈ s

/-- Theorem: Drawing a card numbered 1 from a set of cards numbered 1 to 10 is a random event -/
theorem draw_one_is_random_event : IsRandomEvent DrawOne :=
sorry

end draw_one_is_random_event_l2132_213293


namespace complex_equation_solution_l2132_213219

theorem complex_equation_solution :
  ∀ (a b : ℝ), (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end complex_equation_solution_l2132_213219


namespace arithmetic_geometric_sequence_relation_l2132_213212

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ - a₁ = d ∧ a₁ - (-9) = d ∧ (-1) - a₂ = d

-- Define the geometric sequence
def geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ r : ℝ, b₁ / (-9) = r ∧ b₂ / b₁ = r ∧ b₃ / b₂ = r ∧ (-1) / b₃ = r

-- State the theorem
theorem arithmetic_geometric_sequence_relation :
  ∀ a₁ a₂ b₁ b₂ b₃ : ℝ,
  arithmetic_sequence a₁ a₂ →
  geometric_sequence b₁ b₂ b₃ →
  a₂ * b₂ - a₁ * b₂ = -8 :=
by
  sorry

end arithmetic_geometric_sequence_relation_l2132_213212


namespace infinite_sum_of_squares_with_neighbors_l2132_213237

theorem infinite_sum_of_squares_with_neighbors (k : ℕ) :
  ∃ n : ℕ,
    (∃ a b : ℕ, n = a^2 + b^2) ∧
    (∀ x y : ℕ, (n - 1) ≠ x^2 + y^2) ∧
    (∀ x y : ℕ, (n + 1) ≠ x^2 + y^2) :=
by sorry

end infinite_sum_of_squares_with_neighbors_l2132_213237


namespace sharpshooter_target_orders_l2132_213254

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def multisetPermutations (total : ℕ) (counts : List ℕ) : ℕ :=
  factorial total / (counts.map factorial).prod

theorem sharpshooter_target_orders : 
  let total_targets : ℕ := 8
  let column_targets : List ℕ := [2, 3, 2, 1]
  multisetPermutations total_targets column_targets = 1680 := by
  sorry

end sharpshooter_target_orders_l2132_213254


namespace simplify_expression_l2132_213276

theorem simplify_expression : (5^7 + 3^6) * (1^5 - (-1)^4)^10 = 0 := by
  sorry

end simplify_expression_l2132_213276


namespace even_function_implies_c_eq_neg_four_l2132_213229

/-- Given a function f and a constant c, we define g in terms of f and c. -/
def f (x : ℝ) : ℝ := x^2 + 4*x + 3

def g (c : ℝ) (x : ℝ) : ℝ := f x + c*x

/-- A function h is even if h(-x) = h(x) for all x. -/
def IsEven (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

/-- If g is an even function, then c must equal -4. -/
theorem even_function_implies_c_eq_neg_four :
  IsEven (g c) → c = -4 := by sorry

end even_function_implies_c_eq_neg_four_l2132_213229


namespace mike_arcade_time_mike_play_time_l2132_213255

/-- Given Mike's weekly pay and arcade expenses, calculate his play time in minutes -/
theorem mike_arcade_time (weekly_pay : ℕ) (food_cost : ℕ) (hourly_rate : ℕ) : ℕ :=
  let arcade_budget := weekly_pay / 2
  let token_budget := arcade_budget - food_cost
  let play_hours := token_budget / hourly_rate
  play_hours * 60

/-- Prove that Mike can play for 300 minutes given the specific conditions -/
theorem mike_play_time :
  mike_arcade_time 100 10 8 = 300 := by
  sorry

end mike_arcade_time_mike_play_time_l2132_213255


namespace non_seniors_playing_instrument_l2132_213225

theorem non_seniors_playing_instrument (total_students : ℕ) 
  (senior_play_percent : ℚ) (non_senior_not_play_percent : ℚ) 
  (total_not_play_percent : ℚ) :
  total_students = 500 →
  senior_play_percent = 2/5 →
  non_senior_not_play_percent = 3/10 →
  total_not_play_percent = 234/500 →
  ∃ (seniors non_seniors : ℕ),
    seniors + non_seniors = total_students ∧
    (seniors : ℚ) * (1 - senior_play_percent) + 
    (non_seniors : ℚ) * non_senior_not_play_percent = 
    (total_students : ℚ) * total_not_play_percent ∧
    (non_seniors : ℚ) * (1 - non_senior_not_play_percent) = 154 :=
by sorry

end non_seniors_playing_instrument_l2132_213225


namespace value_of_b_plus_a_l2132_213251

theorem value_of_b_plus_a (a b : ℝ) : 
  (abs a = 8) → 
  (abs b = 2) → 
  (abs (a - b) = b - a) → 
  ((b + a = -6) ∨ (b + a = -10)) := by
sorry

end value_of_b_plus_a_l2132_213251


namespace profit_percent_for_given_ratio_l2132_213208

/-- If the ratio of cost price to selling price is 4:5, then the profit percent is 25% -/
theorem profit_percent_for_given_ratio : 
  ∀ (cp sp : ℝ), cp > 0 → sp > 0 → cp / sp = 4 / 5 → (sp - cp) / cp * 100 = 25 :=
by
  sorry

end profit_percent_for_given_ratio_l2132_213208


namespace min_value_y_l2132_213242

noncomputable def y (x a : ℝ) : ℝ := (Real.exp x - a)^2 + (Real.exp (-x) - a)^2

theorem min_value_y (a : ℝ) (h : a ≠ 0) :
  (∀ x, y x a ≥ (if a ≥ 2 then a^2 - 2 else 2*(a-1)^2)) ∧
  (∃ x, y x a = (if a ≥ 2 then a^2 - 2 else 2*(a-1)^2)) := by
  sorry

end min_value_y_l2132_213242


namespace pentagon_cannot_tile_l2132_213222

/-- Represents a regular polygon --/
inductive RegularPolygon
  | EquilateralTriangle
  | Square
  | Pentagon
  | Hexagon

/-- Calculates the interior angle of a regular polygon with n sides --/
def interiorAngle (n : ℕ) : ℚ :=
  180 - (360 / n)

/-- Checks if a polygon can tile the plane --/
def canTilePlane (p : RegularPolygon) : Prop :=
  match p with
  | RegularPolygon.EquilateralTriangle => (360 / interiorAngle 3).isInt
  | RegularPolygon.Square => (360 / interiorAngle 4).isInt
  | RegularPolygon.Pentagon => (360 / interiorAngle 5).isInt
  | RegularPolygon.Hexagon => (360 / interiorAngle 6).isInt

/-- Theorem stating that only the pentagon cannot tile the plane --/
theorem pentagon_cannot_tile :
  ∀ p : RegularPolygon,
    ¬(canTilePlane p) ↔ p = RegularPolygon.Pentagon :=
by sorry

end pentagon_cannot_tile_l2132_213222


namespace max_value_product_sum_l2132_213245

theorem max_value_product_sum (X Y Z : ℕ) (h : X + Y + Z = 15) :
  (∀ A B C : ℕ, A + B + C = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ A * B * C + A * B + B * C + C * A) →
  X * Y * Z + X * Y + Y * Z + Z * X = 200 :=
by sorry

end max_value_product_sum_l2132_213245


namespace fifteen_power_equals_R_S_power_l2132_213266

theorem fifteen_power_equals_R_S_power (a b : ℤ) (R S : ℝ) 
  (hR : R = 3^a) (hS : S = 5^b) : 15^(a*b) = R^b * S^a := by
  sorry

end fifteen_power_equals_R_S_power_l2132_213266


namespace days_without_calls_is_250_l2132_213277

/-- Represents the frequency of calls from each grandchild -/
def call_frequency₁ : ℕ := 5
def call_frequency₂ : ℕ := 7

/-- Represents the number of days in the year -/
def days_in_year : ℕ := 365

/-- Calculates the number of days without calls -/
def days_without_calls : ℕ :=
  days_in_year - (days_in_year / call_frequency₁ + days_in_year / call_frequency₂ - days_in_year / (call_frequency₁ * call_frequency₂))

/-- Theorem stating that there are 250 days without calls -/
theorem days_without_calls_is_250 : days_without_calls = 250 := by
  sorry

end days_without_calls_is_250_l2132_213277


namespace probability_at_least_one_non_defective_l2132_213202

theorem probability_at_least_one_non_defective (p_defective : ℝ) (h_p : p_defective = 0.3) :
  let p_all_defective := p_defective ^ 3
  let p_at_least_one_non_defective := 1 - p_all_defective
  p_at_least_one_non_defective = 0.973 := by
sorry

end probability_at_least_one_non_defective_l2132_213202


namespace parabola_point_distance_l2132_213214

/-- Given a parabola y² = 4x and a point A on the parabola,
    if the distance from A to the focus is 4,
    then the distance from A to the origin is √21. -/
theorem parabola_point_distance (A : ℝ × ℝ) :
  A.1 ≥ 0 →  -- Ensure x-coordinate is non-negative
  A.2^2 = 4 * A.1 →  -- A is on the parabola
  (A.1 - 1)^2 + A.2^2 = 16 →  -- Distance from A to focus (1, 0) is 4
  A.1^2 + A.2^2 = 21 :=  -- Distance from A to origin is √21
by sorry

end parabola_point_distance_l2132_213214


namespace fraction_most_compliant_l2132_213257

/-- Represents the compliance of an algebraic expression with standard notation -/
inductive AlgebraicCompliance
  | Compliant
  | NonCompliant

/-- Evaluates the compliance of a mixed number with variable expression -/
def mixedNumberWithVariable (n : ℕ) (m : ℕ) (d : ℕ) (x : String) : AlgebraicCompliance :=
  AlgebraicCompliance.NonCompliant

/-- Evaluates the compliance of a fraction expression -/
def fraction (n : String) (d : String) : AlgebraicCompliance :=
  AlgebraicCompliance.Compliant

/-- Evaluates the compliance of an expression with an attached unit -/
def expressionWithUnit (expr : String) (unit : String) : AlgebraicCompliance :=
  AlgebraicCompliance.NonCompliant

/-- Evaluates the compliance of a multiplication expression -/
def multiplicationExpression (x : String) (n : ℕ) : AlgebraicCompliance :=
  AlgebraicCompliance.NonCompliant

/-- Theorem stating that fraction (b/a) is the most compliant with standard algebraic notation -/
theorem fraction_most_compliant :
  fraction "b" "a" = AlgebraicCompliance.Compliant ∧
  mixedNumberWithVariable 1 1 2 "a" = AlgebraicCompliance.NonCompliant ∧
  expressionWithUnit "3a-1" "个" = AlgebraicCompliance.NonCompliant ∧
  multiplicationExpression "a" 3 = AlgebraicCompliance.NonCompliant :=
by sorry

end fraction_most_compliant_l2132_213257
