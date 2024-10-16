import Mathlib

namespace NUMINAMATH_CALUDE_curve_is_line_segment_l2802_280229

-- Define the parametric equations
def x (t : ℝ) : ℝ := 3 * t^2 + 2
def y (t : ℝ) : ℝ := t^2 - 1

-- Define the range of t
def t_range : Set ℝ := {t | 0 ≤ t ∧ t ≤ 5}

-- Define the curve as a set of points
def curve : Set (ℝ × ℝ) := {(x t, y t) | t ∈ t_range}

-- Theorem statement
theorem curve_is_line_segment : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ (p : ℝ × ℝ), p ∈ curve → a * p.1 + b * p.2 + c = 0) ∧
  (∃ (p q : ℝ × ℝ), p ∈ curve ∧ q ∈ curve ∧ p ≠ q ∧
    ∀ (r : ℝ × ℝ), r ∈ curve → ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ r = (1 - t) • p + t • q) :=
sorry

end NUMINAMATH_CALUDE_curve_is_line_segment_l2802_280229


namespace NUMINAMATH_CALUDE_right_triangle_condition_l2802_280276

theorem right_triangle_condition (A B C : ℝ) (h_triangle : A + B + C = Real.pi) 
  (h_condition : Real.sin A * Real.cos B = 1 - Real.cos A * Real.sin B) : C = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l2802_280276


namespace NUMINAMATH_CALUDE_sector_central_angle_l2802_280220

/-- Given a sector with radius 2 and area 4, its central angle (in absolute value) is 2 radians -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = 4) :
  |2 * area / r^2| = 2 :=
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2802_280220


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2802_280228

theorem imaginary_part_of_z (z : ℂ) (h : z = Complex.I * (2 - z)) : z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2802_280228


namespace NUMINAMATH_CALUDE_art_kits_count_l2802_280215

theorem art_kits_count (total_students : ℕ) (students_per_kit : ℕ) 
  (artworks_group1 : ℕ) (artworks_group2 : ℕ) (total_artworks : ℕ) : ℕ :=
  let num_kits := total_students / students_per_kit
  let half_students := total_students / 2
  let artworks_from_group1 := half_students * artworks_group1
  let artworks_from_group2 := half_students * artworks_group2
  by
    have h1 : total_students = 10 := by sorry
    have h2 : students_per_kit = 2 := by sorry
    have h3 : artworks_group1 = 3 := by sorry
    have h4 : artworks_group2 = 4 := by sorry
    have h5 : total_artworks = 35 := by sorry
    have h6 : artworks_from_group1 + artworks_from_group2 = total_artworks := by sorry
    exact num_kits

end NUMINAMATH_CALUDE_art_kits_count_l2802_280215


namespace NUMINAMATH_CALUDE_compound_interest_with_contributions_l2802_280224

theorem compound_interest_with_contributions
  (initial_amount : ℝ)
  (interest_rate : ℝ)
  (annual_contribution : ℝ)
  (years : ℕ)
  (h1 : initial_amount = 76800)
  (h2 : interest_rate = 0.125)
  (h3 : annual_contribution = 5000)
  (h4 : years = 2) :
  let amount_after_first_year := initial_amount * (1 + interest_rate)
  let total_after_first_year := amount_after_first_year + annual_contribution
  let amount_after_second_year := total_after_first_year * (1 + interest_rate)
  let final_amount := amount_after_second_year + annual_contribution
  final_amount = 107825 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_with_contributions_l2802_280224


namespace NUMINAMATH_CALUDE_cat_pictures_count_l2802_280278

/-- Represents the number of photos on Toby's camera roll at different stages -/
structure PhotoCount where
  initial : ℕ
  afterFirstDeletion : ℕ
  final : ℕ

/-- Represents the number of photos deleted or added at different stages -/
structure PhotoChanges where
  firstDeletion : ℕ
  catPictures : ℕ
  friendPhotos : ℕ
  secondDeletion : ℕ

/-- Theorem stating the relationship between cat pictures and friend photos -/
theorem cat_pictures_count (p : PhotoCount) (c : PhotoChanges) :
  p.initial = 63 →
  p.final = 84 →
  c.firstDeletion = 7 →
  c.secondDeletion = 3 →
  p.afterFirstDeletion = p.initial - c.firstDeletion →
  p.final = p.afterFirstDeletion + c.catPictures + c.friendPhotos - c.secondDeletion →
  c.catPictures = 31 - c.friendPhotos := by
  sorry


end NUMINAMATH_CALUDE_cat_pictures_count_l2802_280278


namespace NUMINAMATH_CALUDE_bisection_interval_valid_l2802_280216

-- Define the function f(x) = x^3 + 5
def f (x : ℝ) : ℝ := x^3 + 5

-- Theorem statement
theorem bisection_interval_valid :
  ∃ (a b : ℝ), a = -2 ∧ b = 1 ∧ f a * f b < 0 :=
by sorry

end NUMINAMATH_CALUDE_bisection_interval_valid_l2802_280216


namespace NUMINAMATH_CALUDE_max_parts_five_lines_max_parts_recurrence_l2802_280290

/-- The maximum number of parts a plane can be divided into by n lines -/
def max_parts (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => max_parts m + (m + 1)

/-- Theorem stating the maximum number of parts for 5 lines -/
theorem max_parts_five_lines :
  max_parts 5 = 16 :=
by
  -- The proof goes here
  sorry

/-- Lemma for one line -/
lemma one_line_two_parts :
  max_parts 1 = 2 :=
by
  -- The proof goes here
  sorry

/-- Lemma for two lines -/
lemma two_lines_four_parts :
  max_parts 2 = 4 :=
by
  -- The proof goes here
  sorry

/-- Theorem proving the recurrence relation -/
theorem max_parts_recurrence (n : ℕ) :
  max_parts (n + 1) = max_parts n + (n + 1) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_max_parts_five_lines_max_parts_recurrence_l2802_280290


namespace NUMINAMATH_CALUDE_prob_at_least_half_girls_l2802_280293

/-- The number of children in the family -/
def num_children : ℕ := 5

/-- The probability of having a girl for each child -/
def prob_girl : ℚ := 1/2

/-- The number of possible combinations of boys and girls -/
def total_combinations : ℕ := 2^num_children

/-- The number of combinations with at least half girls -/
def favorable_combinations : ℕ := (num_children.choose 3) + (num_children.choose 4) + (num_children.choose 5)

/-- The probability of having at least half girls in a family of five children -/
theorem prob_at_least_half_girls : 
  (favorable_combinations : ℚ) / total_combinations = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_half_girls_l2802_280293


namespace NUMINAMATH_CALUDE_scatter_plot_for_linear_relationships_l2802_280259

-- Define the concept of a data visualization method
def DataVisualizationMethod : Type := String

-- Define scatter plot as a data visualization method
def scatter_plot : DataVisualizationMethod := "Scatter plot"

-- Define the property of showing relationships between data points
def shows_point_relationships (method : DataVisualizationMethod) : Prop := 
  method = scatter_plot

-- Define the property of being appropriate for determining linear relationships
def appropriate_for_linear_relationships (method : DataVisualizationMethod) : Prop :=
  shows_point_relationships method

-- Theorem stating that scatter plot is appropriate for determining linear relationships
theorem scatter_plot_for_linear_relationships :
  appropriate_for_linear_relationships scatter_plot :=
by
  sorry


end NUMINAMATH_CALUDE_scatter_plot_for_linear_relationships_l2802_280259


namespace NUMINAMATH_CALUDE_equation_roots_imply_m_range_l2802_280225

theorem equation_roots_imply_m_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   4^x₁ - m * 2^(x₁+1) + 2 - m = 0 ∧
   4^x₂ - m * 2^(x₂+1) + 2 - m = 0) →
  1 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_imply_m_range_l2802_280225


namespace NUMINAMATH_CALUDE_projectile_speed_problem_l2802_280274

theorem projectile_speed_problem (initial_distance : ℝ) (second_projectile_speed : ℝ) (time_to_meet : ℝ) :
  initial_distance = 1182 →
  second_projectile_speed = 525 →
  time_to_meet = 1.2 →
  ∃ (first_projectile_speed : ℝ),
    first_projectile_speed = 460 ∧
    (first_projectile_speed + second_projectile_speed) * time_to_meet = initial_distance :=
by sorry

end NUMINAMATH_CALUDE_projectile_speed_problem_l2802_280274


namespace NUMINAMATH_CALUDE_derivative_inequality_l2802_280240

theorem derivative_inequality (a : ℝ) (ha : a > 0) (x : ℝ) (hx : x ≥ 1) :
  let f : ℝ → ℝ := λ x => a * Real.log x + x + 2
  (deriv f) x < x^2 + (a + 2) * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_inequality_l2802_280240


namespace NUMINAMATH_CALUDE_system_solutions_l2802_280203

theorem system_solutions (x₁ x₂ x₃ : ℝ) : 
  (2 * x₁^2 / (1 + x₁^2) = x₂ ∧ 
   2 * x₂^2 / (1 + x₂^2) = x₃ ∧ 
   2 * x₃^2 / (1 + x₃^2) = x₁) ↔ 
  ((x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0) ∨ 
   (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2802_280203


namespace NUMINAMATH_CALUDE_debbie_large_boxes_l2802_280209

def large_box_tape : ℕ := 5
def medium_box_tape : ℕ := 3
def small_box_tape : ℕ := 2
def medium_boxes_packed : ℕ := 8
def small_boxes_packed : ℕ := 5
def total_tape_used : ℕ := 44

theorem debbie_large_boxes :
  ∃ (large_boxes : ℕ),
    large_boxes * large_box_tape +
    medium_boxes_packed * medium_box_tape +
    small_boxes_packed * small_box_tape = total_tape_used ∧
    large_boxes = 2 :=
by sorry

end NUMINAMATH_CALUDE_debbie_large_boxes_l2802_280209


namespace NUMINAMATH_CALUDE_platform_length_l2802_280258

/-- The length of a platform given a train's speed, crossing time, and length -/
theorem platform_length (train_speed : Real) (crossing_time : Real) (train_length : Real) :
  train_speed = 72 * (1000 / 3600) →
  crossing_time = 26 →
  train_length = 240.0416 →
  train_speed * crossing_time - train_length = 279.9584 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2802_280258


namespace NUMINAMATH_CALUDE_ball_path_length_l2802_280233

theorem ball_path_length (A B C M : ℝ × ℝ) : 
  -- Triangle ABC is a right triangle with ∠ABC = 90° and ∠BAC = 60°
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 →
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = (B.1 - A.1)^2 + (B.2 - A.2)^2 →
  -- AB = 6
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 36 →
  -- M is the midpoint of BC
  M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  -- The length of the path from M to AB to AC and back to M is 3√21
  ∃ (P Q : ℝ × ℝ), 
    P.1 = B.1 ∧ 
    (Q.1 - A.1) * (C.1 - A.1) + (Q.2 - A.2) * (C.2 - A.2) = 0 ∧
    (P.1 - M.1)^2 + (P.2 - M.2)^2 + 
    (Q.1 - P.1)^2 + (Q.2 - P.2)^2 + 
    (M.1 - Q.1)^2 + (M.2 - Q.2)^2 = 189 := by
  sorry

end NUMINAMATH_CALUDE_ball_path_length_l2802_280233


namespace NUMINAMATH_CALUDE_only_vegetarian_count_l2802_280235

/-- Represents the number of people in different dietary categories in a family --/
structure FamilyDiet where
  only_nonveg : ℕ
  both_veg_and_nonveg : ℕ
  total_veg : ℕ

/-- Theorem stating the number of people who eat only vegetarian --/
theorem only_vegetarian_count (f : FamilyDiet) 
  (h1 : f.only_nonveg = 8)
  (h2 : f.both_veg_and_nonveg = 6)
  (h3 : f.total_veg = 19) :
  f.total_veg - f.both_veg_and_nonveg = 13 := by
  sorry

#check only_vegetarian_count

end NUMINAMATH_CALUDE_only_vegetarian_count_l2802_280235


namespace NUMINAMATH_CALUDE_billiard_ball_weight_l2802_280286

/-- Given an empty box weighing 0.5 kg and a box containing 6 identical billiard balls
    weighing 1.82 kg, prove that each billiard ball weighs 0.22 kg. -/
theorem billiard_ball_weight (empty_box_weight : ℝ) (full_box_weight : ℝ) :
  empty_box_weight = 0.5 →
  full_box_weight = 1.82 →
  (full_box_weight - empty_box_weight) / 6 = 0.22 := by
  sorry

end NUMINAMATH_CALUDE_billiard_ball_weight_l2802_280286


namespace NUMINAMATH_CALUDE_max_value_of_operation_l2802_280238

theorem max_value_of_operation : ∃ (m : ℕ), 
  (∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 → 3 * (300 - n) ≤ m) ∧ 
  (∃ (n : ℕ), 10 ≤ n ∧ n ≤ 99 ∧ 3 * (300 - n) = m) ∧ 
  m = 870 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_operation_l2802_280238


namespace NUMINAMATH_CALUDE_parabola_chord_length_l2802_280226

/-- The length of a chord AB of a parabola y^2 = 8x intersected by a line y = kx - 2 -/
theorem parabola_chord_length (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2^2 = 8 * A.1 ∧ A.2 = k * A.1 - 2) ∧ 
    (B.2^2 = 8 * B.1 ∧ B.2 = k * B.1 - 2) ∧
    A ≠ B ∧
    (A.1 + B.1) / 2 = 2) →
  ∃ A B : ℝ × ℝ, 
    (A.2^2 = 8 * A.1 ∧ A.2 = k * A.1 - 2) ∧ 
    (B.2^2 = 8 * B.1 ∧ B.2 = k * B.1 - 2) ∧
    A ≠ B ∧
    (A.1 + B.1) / 2 = 2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 15 :=
by sorry


end NUMINAMATH_CALUDE_parabola_chord_length_l2802_280226


namespace NUMINAMATH_CALUDE_right_triangle_cos_b_l2802_280237

theorem right_triangle_cos_b (A B C : ℝ) (h1 : A = 90) (h2 : Real.sin B = 3/5) :
  Real.cos B = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cos_b_l2802_280237


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2802_280208

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 8) * (Real.sqrt 5 / Real.sqrt 9) * (Real.sqrt 7 / Real.sqrt 12) = 
  (35 * Real.sqrt 70) / 840 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2802_280208


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2802_280296

theorem arithmetic_sequence_problem (a b c : ℝ) : 
  (b - a = c - b) →  -- arithmetic sequence condition
  (a + b + c = 9) →  -- sum condition
  (a * b = 6 * c) →  -- product condition
  (a = 4 ∧ b = 3 ∧ c = 2) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2802_280296


namespace NUMINAMATH_CALUDE_largest_package_size_l2802_280267

theorem largest_package_size (juan_markers alicia_markers : ℕ) 
  (h1 : juan_markers = 36) (h2 : alicia_markers = 48) : 
  Nat.gcd juan_markers alicia_markers = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l2802_280267


namespace NUMINAMATH_CALUDE_intersection_product_is_three_l2802_280283

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + 2*x + y^2 + 4*y + 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + 4*x + y^2 + 4*y + 7 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (-1.5, -2)

-- Theorem statement
theorem intersection_product_is_three :
  let (x, y) := intersection_point
  circle1 x y ∧ circle2 x y ∧ x * y = 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_product_is_three_l2802_280283


namespace NUMINAMATH_CALUDE_range_of_f_l2802_280204

def f (x : ℝ) : ℝ := 2 * x - 1

theorem range_of_f : 
  ∀ y ∈ Set.Icc (-1 : ℝ) 3, ∃ x ∈ Set.Icc 0 2, f x = y ∧
  ∀ x ∈ Set.Icc 0 2, f x ∈ Set.Icc (-1 : ℝ) 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_f_l2802_280204


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l2802_280292

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem tenth_term_of_arithmetic_sequence 
  (a₁ : ℚ) (a₂₀ : ℚ) (h₁ : a₁ = 5/11) (h₂₀ : a₂₀ = 9/11)
  (h_seq : ∀ n, arithmetic_sequence a₁ ((a₂₀ - a₁) / 19) n = 
    arithmetic_sequence (5/11) ((9/11 - 5/11) / 19) n) :
  arithmetic_sequence a₁ ((a₂₀ - a₁) / 19) 10 = 1233/2309 :=
by sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l2802_280292


namespace NUMINAMATH_CALUDE_problem_statement_l2802_280201

theorem problem_statement (a : ℝ) (h : 2 * a - 1 / a = 3) : 16 * a^4 + 1 / a^4 = 161 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2802_280201


namespace NUMINAMATH_CALUDE_equal_percentage_price_l2802_280210

/-- Represents the cost and various selling prices of an article -/
structure Article where
  cp : ℝ  -- Cost price
  sp_profit : ℝ  -- Selling price with 25% profit
  sp_loss : ℝ  -- Selling price with loss
  sp_equal : ℝ  -- Selling price where % profit = % loss

/-- Conditions for the article pricing problem -/
def article_conditions (a : Article) : Prop :=
  a.sp_profit = a.cp * 1.25 ∧  -- 25% profit condition
  a.sp_profit = 1625 ∧
  a.sp_loss = 1280 ∧
  a.sp_loss < a.cp  -- Ensures sp_loss results in a loss

/-- Theorem stating the selling price where percentage profit equals percentage loss -/
theorem equal_percentage_price (a : Article) 
  (h : article_conditions a) : a.sp_equal = 1320 := by
  sorry

#check equal_percentage_price

end NUMINAMATH_CALUDE_equal_percentage_price_l2802_280210


namespace NUMINAMATH_CALUDE_rudolph_trip_signs_per_mile_l2802_280245

/-- Rudolph's car trip across town -/
def rudolph_trip (miles_base : ℕ) (miles_extra : ℕ) (signs_base : ℕ) (signs_less : ℕ) : ℚ :=
  let total_miles : ℕ := miles_base + miles_extra
  let total_signs : ℕ := signs_base - signs_less
  (total_signs : ℚ) / (total_miles : ℚ)

/-- Theorem stating the number of stop signs per mile Rudolph encountered -/
theorem rudolph_trip_signs_per_mile :
  rudolph_trip 5 2 17 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_rudolph_trip_signs_per_mile_l2802_280245


namespace NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l2802_280260

theorem largest_n_satisfying_conditions : ∃ (m k : ℤ),
  181^2 = (m + 1)^3 - m^3 ∧
  2 * 181 + 79 = k^2 ∧
  ∀ (n : ℤ), n > 181 → ¬(∃ (m' k' : ℤ), n^2 = (m' + 1)^3 - m'^3 ∧ 2 * n + 79 = k'^2) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l2802_280260


namespace NUMINAMATH_CALUDE_median_is_eight_l2802_280213

-- Define the daily production values and the number of workers for each value
def daily_production : List ℕ := [5, 6, 7, 8, 9, 10]
def worker_count : List ℕ := [4, 5, 8, 9, 6, 4]

-- Define a function to calculate the median
def median (production : List ℕ) (workers : List ℕ) : ℚ :=
  sorry

-- Theorem statement
theorem median_is_eight :
  median daily_production worker_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_median_is_eight_l2802_280213


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l2802_280250

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem pirate_treasure_distribution : 
  ∃ (x : ℕ), 
    x > 0 ∧ 
    sum_of_first_n x = 3 * x ∧ 
    x + 3 * x = 20 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l2802_280250


namespace NUMINAMATH_CALUDE_isosceles_side_in_equilateral_l2802_280239

/-- The length of a side of an isosceles triangle inscribed in an equilateral triangle -/
theorem isosceles_side_in_equilateral (s : ℝ) (h : s = 2) :
  let equilateral_side := s
  let isosceles_base := equilateral_side / 2
  let isosceles_side := Real.sqrt (7 / 3)
  ∃ (triangle : Set (ℝ × ℝ)),
    (∀ p ∈ triangle, p.1 ≥ 0 ∧ p.1 ≤ equilateral_side ∧ p.2 ≥ 0 ∧ p.2 ≤ equilateral_side * Real.sqrt 3 / 2) ∧
    (∃ (a b c : ℝ × ℝ), a ∈ triangle ∧ b ∈ triangle ∧ c ∈ triangle ∧
      (a.1 - b.1)^2 + (a.2 - b.2)^2 = equilateral_side^2 ∧
      (b.1 - c.1)^2 + (b.2 - c.2)^2 = equilateral_side^2 ∧
      (c.1 - a.1)^2 + (c.2 - a.2)^2 = equilateral_side^2) ∧
    (∃ (p q r : ℝ × ℝ), p ∈ triangle ∧ q ∈ triangle ∧ r ∈ triangle ∧
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = isosceles_side^2 ∧
      (q.1 - r.1)^2 + (q.2 - r.2)^2 = isosceles_side^2 ∧
      (r.1 - p.1)^2 + (r.2 - p.2)^2 = isosceles_base^2) := by
  sorry


end NUMINAMATH_CALUDE_isosceles_side_in_equilateral_l2802_280239


namespace NUMINAMATH_CALUDE_second_integer_value_l2802_280212

/-- Given three consecutive odd integers where the sum of the first and third is 144,
    prove that the second integer is 72. -/
theorem second_integer_value (a b c : ℤ) : 
  (∃ n : ℤ, a = n - 2 ∧ b = n ∧ c = n + 2) →  -- consecutive odd integers
  (a + c = 144) →                            -- sum of first and third is 144
  b = 72 :=                                  -- second integer is 72
by sorry

end NUMINAMATH_CALUDE_second_integer_value_l2802_280212


namespace NUMINAMATH_CALUDE_english_marks_proof_l2802_280280

def average (numbers : List ℕ) : ℚ :=
  (numbers.sum : ℚ) / numbers.length

theorem english_marks_proof (marks : List ℕ) (h1 : marks.length = 5) 
  (h2 : average marks = 76) 
  (h3 : 69 ∈ marks) (h4 : 92 ∈ marks) (h5 : 64 ∈ marks) (h6 : 82 ∈ marks) : 
  73 ∈ marks := by
  sorry

#check english_marks_proof

end NUMINAMATH_CALUDE_english_marks_proof_l2802_280280


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l2802_280242

theorem smallest_n_for_inequality : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (a : Fin n → ℝ), (∀ i, 1 < a i ∧ a i < 1000) → (∀ i j, i ≠ j → a i ≠ a j) → 
    ∃ i j, i ≠ j ∧ 0 < a i - a j ∧ a i - a j < 1 + 3 * Real.rpow (a i * a j) (1/3)) ∧
  (∀ m : ℕ, m < n → 
    ∃ (a : Fin m → ℝ), (∀ i, 1 < a i ∧ a i < 1000) ∧ (∀ i j, i ≠ j → a i ≠ a j) ∧
      ∀ i j, i ≠ j → ¬(0 < a i - a j ∧ a i - a j < 1 + 3 * Real.rpow (a i * a j) (1/3))) ∧
  n = 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l2802_280242


namespace NUMINAMATH_CALUDE_number_subtraction_problem_l2802_280251

theorem number_subtraction_problem (x : ℝ) : 
  0.30 * x - 70 = 20 → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_problem_l2802_280251


namespace NUMINAMATH_CALUDE_prob_b_greater_a_value_l2802_280262

/-- The number of possible choices for each person -/
def n : ℕ := 1000

/-- The probability of B picking a number greater than A -/
def prob_b_greater_a : ℚ :=
  (n * (n - 1) / 2) / (n * n)

/-- Theorem: The probability of B picking a number greater than A is 499500/1000000 -/
theorem prob_b_greater_a_value : prob_b_greater_a = 499500 / 1000000 := by
  sorry

end NUMINAMATH_CALUDE_prob_b_greater_a_value_l2802_280262


namespace NUMINAMATH_CALUDE_inequality_proof_l2802_280222

theorem inequality_proof (a b x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x / (a*y + b*z)) + (y / (a*z + b*x)) + (z / (a*x + b*y)) ≥ 3 / (a + b) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2802_280222


namespace NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l2802_280256

-- System 1
theorem system_1_solution :
  ∃! (x y : ℝ), 3 * x + 2 * y = 8 ∧ y = 2 * x - 3 ∧ x = 2 ∧ y = 1 := by sorry

-- System 2
theorem system_2_solution :
  ∃! (x y : ℝ), 3 * x + 2 * y = 7 ∧ 6 * x - 2 * y = 11 ∧ x = 2 ∧ y = 1/2 := by sorry

end NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l2802_280256


namespace NUMINAMATH_CALUDE_final_value_of_A_l2802_280264

-- Define the initial value of A
def A_initial : ℤ := 15

-- Define the operation as a function
def operation (x : ℤ) : ℤ := -x + 5

-- Theorem stating the final value of A after the operation
theorem final_value_of_A : operation A_initial = -10 := by
  sorry

end NUMINAMATH_CALUDE_final_value_of_A_l2802_280264


namespace NUMINAMATH_CALUDE_probability_sum_10_l2802_280284

def die_faces : Nat := 6

def total_outcomes : Nat := die_faces ^ 3

def favorable_outcomes : Nat := 30

theorem probability_sum_10 : 
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_10_l2802_280284


namespace NUMINAMATH_CALUDE_initial_distance_between_cars_l2802_280221

/-- 
Given two cars A and B traveling in the same direction:
- Car A's speed is 58 mph
- Car B's speed is 50 mph
- After 6 hours, Car A is 8 miles ahead of Car B
Prove that the initial distance between Car A and Car B is 40 miles
-/
theorem initial_distance_between_cars (speed_A speed_B time_elapsed final_distance : ℝ) 
  (h1 : speed_A = 58)
  (h2 : speed_B = 50)
  (h3 : time_elapsed = 6)
  (h4 : final_distance = 8) :
  speed_A * time_elapsed - speed_B * time_elapsed - final_distance = 40 :=
by sorry

end NUMINAMATH_CALUDE_initial_distance_between_cars_l2802_280221


namespace NUMINAMATH_CALUDE_sin_sum_product_identity_l2802_280241

theorem sin_sum_product_identity : 
  Real.sin (17 * π / 180) * Real.sin (223 * π / 180) + 
  Real.sin (253 * π / 180) * Real.sin (313 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_product_identity_l2802_280241


namespace NUMINAMATH_CALUDE_three_equal_perimeter_triangles_l2802_280205

def stick_lengths : List ℕ := [2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 9]

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def forms_triangle (lengths : List ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ lengths ∧ b ∈ lengths ∧ c ∈ lengths ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  is_triangle a b c ∧
  a + b + c = 14

theorem three_equal_perimeter_triangles :
  ∃ (t1 t2 t3 : List ℕ),
    t1 ⊆ stick_lengths ∧
    t2 ⊆ stick_lengths ∧
    t3 ⊆ stick_lengths ∧
    t1 ∩ t2 = ∅ ∧ t2 ∩ t3 = ∅ ∧ t3 ∩ t1 = ∅ ∧
    forms_triangle t1 ∧
    forms_triangle t2 ∧
    forms_triangle t3 :=
  sorry

end NUMINAMATH_CALUDE_three_equal_perimeter_triangles_l2802_280205


namespace NUMINAMATH_CALUDE_inequality_solution_l2802_280294

theorem inequality_solution (x : ℝ) :
  (1 / (x^2 + 1) > 5/x + 21/10) ↔ -2 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2802_280294


namespace NUMINAMATH_CALUDE_lollipops_left_l2802_280289

theorem lollipops_left (initial : ℕ) (eaten : ℕ) (h1 : initial = 12) (h2 : eaten = 5) :
  initial - eaten = 7 := by
  sorry

end NUMINAMATH_CALUDE_lollipops_left_l2802_280289


namespace NUMINAMATH_CALUDE_chair_color_probability_l2802_280254

/-- The probability that the last two remaining chairs are of the same color -/
def same_color_probability (black_chairs brown_chairs : ℕ) : ℚ :=
  let total_chairs := black_chairs + brown_chairs
  let black_prob := (black_chairs : ℚ) / total_chairs * ((black_chairs - 1) : ℚ) / (total_chairs - 1)
  let brown_prob := (brown_chairs : ℚ) / total_chairs * ((brown_chairs - 1) : ℚ) / (total_chairs - 1)
  black_prob + brown_prob

/-- Theorem stating that the probability of the last two chairs being the same color is 43/88 -/
theorem chair_color_probability :
  same_color_probability 15 18 = 43 / 88 := by
  sorry

end NUMINAMATH_CALUDE_chair_color_probability_l2802_280254


namespace NUMINAMATH_CALUDE_special_triangle_area_l2802_280236

/-- A triangle with specific properties -/
structure SpecialTriangle where
  -- The angle between the two longest sides (in degrees)
  x : ℝ
  -- The perimeter of the triangle (in cm)
  perimeter : ℝ
  -- The inradius of the triangle (in cm)
  inradius : ℝ
  -- Constraint on the angle x
  angle_constraint : 60 < x ∧ x < 120
  -- Given perimeter value
  perimeter_value : perimeter = 48
  -- Given inradius value
  inradius_value : inradius = 2.5

/-- The area of a triangle given its perimeter and inradius -/
def triangleArea (t : SpecialTriangle) : ℝ := t.perimeter * t.inradius

/-- Theorem stating that the area of the special triangle is 120 cm² -/
theorem special_triangle_area (t : SpecialTriangle) : triangleArea t = 120 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_area_l2802_280236


namespace NUMINAMATH_CALUDE_fraction_sum_and_complex_fraction_l2802_280272

theorem fraction_sum_and_complex_fraction (a b m : ℝ) 
  (h1 : a ≠ b) (h2 : m ≠ 1) (h3 : m ≠ 2) : 
  (a / (a - b) + b / (b - a) = 1) ∧ 
  ((m^2 - 4) / (4 + 4*m + m^2) / ((m - 2) / (2*m - 2)) * ((m + 2) / (m - 1)) = 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_and_complex_fraction_l2802_280272


namespace NUMINAMATH_CALUDE_complex_problem_l2802_280270

-- Define a complex number z
variable (z : ℂ)

-- Define the property of being purely imaginary
def isPurelyImaginary (w : ℂ) : Prop := w.re = 0 ∧ w.im ≠ 0

-- State the theorem
theorem complex_problem :
  isPurelyImaginary z →
  isPurelyImaginary ((z + 2)^2 - 8*I) →
  z = -2*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_problem_l2802_280270


namespace NUMINAMATH_CALUDE_problem_solution_l2802_280247

theorem problem_solution (x z : ℝ) (h1 : x ≠ 0) (h2 : x/3 = z^2) (h3 : x/5 = 5*z) : x = 625/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2802_280247


namespace NUMINAMATH_CALUDE_embankment_completion_time_l2802_280263

/-- The time required for a group of workers to complete an embankment -/
def embankment_time (workers : ℕ) (portion : ℚ) (days : ℚ) : Prop :=
  ∃ (rate : ℚ), rate > 0 ∧ portion = (workers : ℚ) * rate * days

theorem embankment_completion_time :
  embankment_time 60 (1/2) 5 →
  embankment_time 80 1 (15/2) :=
by sorry

end NUMINAMATH_CALUDE_embankment_completion_time_l2802_280263


namespace NUMINAMATH_CALUDE_salary_restoration_l2802_280202

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : original_salary > 0) :
  let reduced_salary := original_salary * (1 - 0.2)
  reduced_salary * (1 + 0.25) = original_salary :=
by sorry

end NUMINAMATH_CALUDE_salary_restoration_l2802_280202


namespace NUMINAMATH_CALUDE_park_population_l2802_280211

theorem park_population (lions leopards elephants zebras : ℕ) : 
  lions = 200 →
  lions = 2 * leopards →
  elephants = (lions + leopards) / 2 →
  zebras = elephants + leopards →
  lions + leopards + elephants + zebras = 700 := by
sorry

end NUMINAMATH_CALUDE_park_population_l2802_280211


namespace NUMINAMATH_CALUDE_negative_of_negative_is_positive_l2802_280234

theorem negative_of_negative_is_positive (y : ℝ) (h : y < 0) : -y > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_of_negative_is_positive_l2802_280234


namespace NUMINAMATH_CALUDE_system_solutions_correct_l2802_280265

theorem system_solutions_correct :
  -- System (1)
  let x₁ := 1
  let y₁ := 2
  -- System (2)
  let x₂ := (1 : ℚ) / 2
  let y₂ := 5
  -- Prove that these solutions satisfy the equations
  (x₁ = 5 - 2 * y₁ ∧ 3 * x₁ - y₁ = 1) ∧
  (2 * x₂ - y₂ = -4 ∧ 4 * x₂ - 5 * y₂ = -23) := by
  sorry


end NUMINAMATH_CALUDE_system_solutions_correct_l2802_280265


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l2802_280253

/-- If three lines intersect at one point, then a specific value of a is determined -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃! p : ℝ × ℝ, (a * p.1 + 2 * p.2 + 8 = 0) ∧ 
                  (4 * p.1 + 3 * p.2 - 10 = 0) ∧ 
                  (2 * p.1 - p.2 = 0)) → 
  a = -12 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l2802_280253


namespace NUMINAMATH_CALUDE_f_non_monotonic_iff_a_in_range_l2802_280232

-- Define the piecewise function f(x)
noncomputable def f (a t x : ℝ) : ℝ :=
  if x ≤ t then (4*a - 3)*x + 2*a - 4 else 2*x^3 - 6*x

-- Define the property of being non-monotonic on ℝ
def is_non_monotonic (f : ℝ → ℝ) : Prop :=
  ∃ x y z : ℝ, x < y ∧ y < z ∧ (f x < f y ∧ f y > f z ∨ f x > f y ∧ f y < f z)

-- State the theorem
theorem f_non_monotonic_iff_a_in_range :
  (∀ t : ℝ, is_non_monotonic (f a t)) ↔ a ∈ Set.Iic (3/4) :=
sorry

end NUMINAMATH_CALUDE_f_non_monotonic_iff_a_in_range_l2802_280232


namespace NUMINAMATH_CALUDE_lighthouse_min_fuel_l2802_280295

/-- Represents the lighthouse generator's operation parameters -/
structure LighthouseGenerator where
  fuel_per_hour : ℝ
  startup_fuel : ℝ
  total_hours : ℝ
  max_stop_time : ℝ
  min_run_time : ℝ

/-- Calculates the minimum fuel required for the lighthouse generator -/
def min_fuel_required (g : LighthouseGenerator) : ℝ :=
  -- The actual calculation would go here
  sorry

/-- Theorem stating the minimum fuel required for the given parameters -/
theorem lighthouse_min_fuel :
  let g : LighthouseGenerator := {
    fuel_per_hour := 6,
    startup_fuel := 0.5,
    total_hours := 10,
    max_stop_time := 1/6,  -- 10 minutes in hours
    min_run_time := 1/4    -- 15 minutes in hours
  }
  min_fuel_required g = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_lighthouse_min_fuel_l2802_280295


namespace NUMINAMATH_CALUDE_ab_gt_ac_l2802_280279

theorem ab_gt_ac (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_ab_gt_ac_l2802_280279


namespace NUMINAMATH_CALUDE_production_days_l2802_280217

theorem production_days (n : ℕ+) 
  (h1 : (n * 50 : ℝ) / n = 50)
  (h2 : ((n * 50 + 90 : ℝ) / (n + 1) = 58)) : 
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l2802_280217


namespace NUMINAMATH_CALUDE_books_per_shelf_l2802_280230

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) 
  (h1 : total_books = 504) (h2 : num_shelves = 9) :
  total_books / num_shelves = 56 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l2802_280230


namespace NUMINAMATH_CALUDE_bob_sister_time_relation_l2802_280223

/-- Bob's current time for a mile in seconds -/
def bob_current_time : ℝ := 640

/-- The percentage improvement Bob needs to make -/
def improvement_percentage : ℝ := 9.062499999999996

/-- Bob's sister's time for a mile in seconds -/
def sister_time : ℝ := 582

/-- Theorem stating the relationship between Bob's current time, improvement percentage, and his sister's time -/
theorem bob_sister_time_relation :
  sister_time = bob_current_time * (1 - improvement_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_bob_sister_time_relation_l2802_280223


namespace NUMINAMATH_CALUDE_inscribed_circle_right_triangle_l2802_280269

theorem inscribed_circle_right_triangle 
  (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_perimeter : a + b + c = 30) 
  (h_tangency_ratio : ∃ (r : ℝ), a = 5*r/2 ∧ b = 12*r/5) : 
  (a = 5 ∧ b = 12 ∧ c = 13) := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_right_triangle_l2802_280269


namespace NUMINAMATH_CALUDE_two_from_three_l2802_280273

/-- The number of combinations of k items from a set of n items -/
def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: There are 3 ways to choose 2 items from a set of 3 items -/
theorem two_from_three : combinations 3 2 = 3 := by sorry

end NUMINAMATH_CALUDE_two_from_three_l2802_280273


namespace NUMINAMATH_CALUDE_library_books_end_of_month_l2802_280206

theorem library_books_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) : 
  initial_books = 75 → 
  loaned_books = 50 → 
  return_rate = 70 / 100 → 
  initial_books - (loaned_books - (return_rate * loaned_books).floor) = 60 := by
sorry

end NUMINAMATH_CALUDE_library_books_end_of_month_l2802_280206


namespace NUMINAMATH_CALUDE_kindergarten_allergies_l2802_280252

/-- Given a kindergarten with the following conditions:
  - T is the total number of children
  - Half of the children are allergic to peanuts
  - 10 children are not allergic to cashew nuts
  - 10 children are allergic to both peanuts and cashew nuts
  - Some children are allergic to cashew nuts
Prove that the number of children not allergic to peanuts and not allergic to cashew nuts is 10 -/
theorem kindergarten_allergies (T : ℕ) : 
  T > 0 →
  T / 2 = (T - T / 2) → -- Half of the children are allergic to peanuts
  ∃ (cashew_allergic : ℕ), cashew_allergic > 0 ∧ cashew_allergic < T → -- Some children are allergic to cashew nuts
  10 = T - cashew_allergic → -- 10 children are not allergic to cashew nuts
  10 ≤ T / 2 → -- 10 children are allergic to both peanuts and cashew nuts
  10 = T - (T / 2 + cashew_allergic - 10) -- Number of children not allergic to peanuts and not allergic to cashew nuts
  := by sorry

end NUMINAMATH_CALUDE_kindergarten_allergies_l2802_280252


namespace NUMINAMATH_CALUDE_triangle_side_a_value_l2802_280277

noncomputable def triangle_side_a (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Define the triangle ABC
  (0 < A ∧ 0 < B ∧ 0 < C) ∧
  (A + B + C = Real.pi) ∧
  -- Relate sides to angles using sine law
  (a / (Real.sin A) = b / (Real.sin B)) ∧
  (b / (Real.sin B) = c / (Real.sin C)) ∧
  -- Given conditions
  (Real.sin B = 3/5) ∧
  (b = 5) ∧
  (A = 2 * B) ∧
  -- Conclusion
  (a = 8)

theorem triangle_side_a_value :
  ∀ (A B C : ℝ) (a b c : ℝ),
  triangle_side_a A B C a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_side_a_value_l2802_280277


namespace NUMINAMATH_CALUDE_initial_tourists_l2802_280243

theorem initial_tourists (T : ℕ) : 
  (T : ℚ) - 2 - (3/7) * ((T : ℚ) - 2) = 16 → T = 30 := by
  sorry

end NUMINAMATH_CALUDE_initial_tourists_l2802_280243


namespace NUMINAMATH_CALUDE_pizza_delivery_time_l2802_280207

theorem pizza_delivery_time (total_pizzas : ℕ) (double_order_stops : ℕ) (time_per_stop : ℕ) : 
  total_pizzas = 12 →
  double_order_stops = 2 →
  time_per_stop = 4 →
  (total_pizzas - 2 * double_order_stops + double_order_stops) * time_per_stop = 40 :=
by sorry

end NUMINAMATH_CALUDE_pizza_delivery_time_l2802_280207


namespace NUMINAMATH_CALUDE_train_passenger_count_l2802_280271

/-- Given a train journey with specific passenger changes, prove the initial number of passengers. -/
theorem train_passenger_count (P : ℕ) : 
  (((P - (P / 3) + 280) / 2 + 12) = 242) → P = 270 := by
  sorry

end NUMINAMATH_CALUDE_train_passenger_count_l2802_280271


namespace NUMINAMATH_CALUDE_strawberry_sales_chloe_strawberry_sales_l2802_280244

/-- Calculates the number of dozens of strawberries sold given the cost per dozen,
    selling price per half dozen, and total profit. -/
theorem strawberry_sales 
  (cost_per_dozen : ℚ) 
  (selling_price_per_half_dozen : ℚ) 
  (total_profit : ℚ) : ℚ :=
  let profit_per_half_dozen := selling_price_per_half_dozen - cost_per_dozen / 2
  let half_dozens_sold := total_profit / profit_per_half_dozen
  let dozens_sold := half_dozens_sold / 2
  dozens_sold

/-- Proves that given the specified conditions, Chloe sold 50 dozens of strawberries. -/
theorem chloe_strawberry_sales : 
  strawberry_sales 50 30 500 = 50 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_sales_chloe_strawberry_sales_l2802_280244


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2802_280268

theorem complex_fraction_sum (a b : ℝ) :
  (Complex.I + 1) / (1 - Complex.I) = Complex.mk a b →
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2802_280268


namespace NUMINAMATH_CALUDE_find_constant_b_l2802_280291

theorem find_constant_b (a b c : ℚ) : 
  (∀ x : ℚ, (4 * x^3 - 3 * x + 7/2) * (a * x^2 + b * x + c) = 
    12 * x^5 - 14 * x^4 + 18 * x^3 - (23/3) * x^2 + (14/2) * x - 3) →
  b = -7/2 := by
sorry

end NUMINAMATH_CALUDE_find_constant_b_l2802_280291


namespace NUMINAMATH_CALUDE_area_between_curves_l2802_280297

theorem area_between_curves : 
  let f (x : ℝ) := x^2
  let g (x : ℝ) := x^3
  ∫ x in (0)..(1), (f x - g x) = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_area_between_curves_l2802_280297


namespace NUMINAMATH_CALUDE_williams_children_probability_l2802_280281

theorem williams_children_probability :
  let n : ℕ := 8  -- number of children
  let p : ℚ := 1/2  -- probability of each child being a boy (or girl)
  let total_outcomes : ℕ := 2^n  -- total number of possible gender combinations
  let balanced_outcomes : ℕ := n.choose (n/2)  -- number of combinations with equal boys and girls
  
  (total_outcomes - balanced_outcomes : ℚ) / total_outcomes = 93/128 :=
by sorry

end NUMINAMATH_CALUDE_williams_children_probability_l2802_280281


namespace NUMINAMATH_CALUDE_problem_solution_l2802_280288

noncomputable section

def f (x : ℝ) : ℝ := Real.log x + 1 / x
def g (x : ℝ) : ℝ := x - Real.log x

theorem problem_solution :
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 → f x ≥ a ∧ 
    ∀ (b : ℝ), (∀ (y : ℝ), y > 0 → f y ≥ b) → b ≤ a) ∧
  (∀ (x : ℝ), x > 1 → f x < g x) ∧
  (∀ (x₁ x₂ : ℝ), x₁ > x₂ ∧ x₂ > 0 ∧ g x₁ = g x₂ → x₁ * x₂ < 1) :=
by sorry

end

end NUMINAMATH_CALUDE_problem_solution_l2802_280288


namespace NUMINAMATH_CALUDE_product_of_ratios_l2802_280218

theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2007 ∧ y₁^3 - 3*x₁^2*y₁ = 2006)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2007 ∧ y₂^3 - 3*x₂^2*y₂ = 2006)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2007 ∧ y₃^3 - 3*x₃^2*y₃ = 2006) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -2011/2006 := by
sorry

end NUMINAMATH_CALUDE_product_of_ratios_l2802_280218


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2802_280282

-- Define the quadratic inequality
def quadratic_inequality (x : ℝ) : Prop := (x - 2) * (x + 2) < 5

-- Define the solution set
def solution_set : Set ℝ := {x | -3 < x ∧ x < 3}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | quadratic_inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2802_280282


namespace NUMINAMATH_CALUDE_leftover_coins_value_l2802_280255

def roll_size : ℕ := 40
def quarter_value : ℚ := 0.25
def nickel_value : ℚ := 0.05

def mia_quarters : ℕ := 92
def mia_nickels : ℕ := 184
def thomas_quarters : ℕ := 138
def thomas_nickels : ℕ := 212

def total_quarters : ℕ := mia_quarters + thomas_quarters
def total_nickels : ℕ := mia_nickels + thomas_nickels

def leftover_quarters : ℕ := total_quarters % roll_size
def leftover_nickels : ℕ := total_nickels % roll_size

def leftover_value : ℚ := leftover_quarters * quarter_value + leftover_nickels * nickel_value

theorem leftover_coins_value : leftover_value = 9.30 := by
  sorry

end NUMINAMATH_CALUDE_leftover_coins_value_l2802_280255


namespace NUMINAMATH_CALUDE_apple_pies_count_l2802_280246

theorem apple_pies_count (pecan_pies : ℕ) (total_rows : ℕ) (pies_per_row : ℕ) : 
  pecan_pies = 16 →
  total_rows = 6 →
  pies_per_row = 5 →
  ∃ (apple_pies : ℕ), apple_pies = total_rows * pies_per_row - pecan_pies ∧ apple_pies = 14 := by
  sorry

end NUMINAMATH_CALUDE_apple_pies_count_l2802_280246


namespace NUMINAMATH_CALUDE_power_equation_exponent_l2802_280214

theorem power_equation_exponent (x : ℝ) (n : ℝ) (h : x ≠ 0) : 
  x^3 / x = x^n → n = 2 :=
by sorry

end NUMINAMATH_CALUDE_power_equation_exponent_l2802_280214


namespace NUMINAMATH_CALUDE_tens_digit_of_sum_is_one_l2802_280275

/-- 
Theorem: For any three-digit number where the hundreds digit is 3 more than the units digit,
the tens digit of the sum of this number and its reverse is always 1.
-/
theorem tens_digit_of_sum_is_one (c b : ℕ) (h1 : c < 10) (h2 : b < 10) : 
  (((202 * c + 20 * b + 303) / 10) % 10) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_sum_is_one_l2802_280275


namespace NUMINAMATH_CALUDE_hearts_to_diamonds_ratio_l2802_280257

/-- Represents the number of cards of each suit in a player's hand -/
structure CardCounts where
  spades : ℕ
  diamonds : ℕ
  hearts : ℕ
  clubs : ℕ

/-- The conditions of the card counting problem -/
def validCardCounts (c : CardCounts) : Prop :=
  c.spades + c.diamonds + c.hearts + c.clubs = 13 ∧
  c.spades + c.clubs = 7 ∧
  c.diamonds + c.hearts = 6 ∧
  c.diamonds = 2 * c.spades ∧
  c.clubs = 6

theorem hearts_to_diamonds_ratio (c : CardCounts) 
  (h : validCardCounts c) : c.hearts = 2 * c.diamonds := by
  sorry

end NUMINAMATH_CALUDE_hearts_to_diamonds_ratio_l2802_280257


namespace NUMINAMATH_CALUDE_sqrt_identity_l2802_280227

theorem sqrt_identity : (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2)^2 = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_identity_l2802_280227


namespace NUMINAMATH_CALUDE_dresser_contents_l2802_280299

/-- Given a dresser with pants, shorts, and shirts in the ratio 7 : 7 : 10,
    prove that if there are 14 pants, there are 20 shirts. -/
theorem dresser_contents (pants shorts shirts : ℕ) : 
  pants = 14 →
  pants * 10 = shirts * 7 →
  shirts = 20 := by
  sorry

end NUMINAMATH_CALUDE_dresser_contents_l2802_280299


namespace NUMINAMATH_CALUDE_negation_equivalence_l2802_280287

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 1 ≤ 0) ↔ (∀ x : ℝ, x > 0 → x^2 - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2802_280287


namespace NUMINAMATH_CALUDE_davids_homework_time_l2802_280266

theorem davids_homework_time (math_time spelling_time reading_time : ℕ) 
  (h1 : math_time = 15)
  (h2 : spelling_time = 18)
  (h3 : reading_time = 27) :
  math_time + spelling_time + reading_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_davids_homework_time_l2802_280266


namespace NUMINAMATH_CALUDE_total_seashells_l2802_280231

theorem total_seashells (red_shells green_shells other_shells : ℕ) 
  (h1 : red_shells = 76)
  (h2 : green_shells = 49)
  (h3 : other_shells = 166) :
  red_shells + green_shells + other_shells = 291 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l2802_280231


namespace NUMINAMATH_CALUDE_four_digit_number_divisible_by_twelve_l2802_280219

theorem four_digit_number_divisible_by_twelve (n : ℕ) (A : ℕ) : 
  n = 2000 + 10 * A + 2 →
  A < 10 →
  n % 12 = 0 →
  n = 2052 := by
sorry

end NUMINAMATH_CALUDE_four_digit_number_divisible_by_twelve_l2802_280219


namespace NUMINAMATH_CALUDE_min_obtuse_angles_convex_octagon_l2802_280285

/-- A convex octagon -/
structure ConvexOctagon where
  -- We don't need to define the structure explicitly for this problem

/-- The number of angles in an octagon -/
def num_angles : ℕ := 8

/-- The sum of exterior angles in any polygon -/
def sum_exterior_angles : ℕ := 360

/-- Theorem: In a convex octagon, the minimum number of obtuse interior angles is 5 -/
theorem min_obtuse_angles_convex_octagon (O : ConvexOctagon) : 
  ∃ (n : ℕ), n ≥ 5 ∧ n = (num_angles - (sum_exterior_angles / 90)) := by
  sorry

end NUMINAMATH_CALUDE_min_obtuse_angles_convex_octagon_l2802_280285


namespace NUMINAMATH_CALUDE_positive_real_inequality_l2802_280261

theorem positive_real_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * (a^2 + b*c)) / (b + c) + (b * (b^2 + c*a)) / (c + a) + (c * (c^2 + a*b)) / (a + b) ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l2802_280261


namespace NUMINAMATH_CALUDE_inverse_proposition_l2802_280248

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x < 0 → x^2 > 0

-- Define the inverse proposition
def inverse_prop (x : ℝ) : Prop := ¬(x^2 > 0) → ¬(x < 0)

-- Theorem stating that inverse_prop is the inverse of original_prop
theorem inverse_proposition :
  (∀ x : ℝ, original_prop x) ↔ (∀ x : ℝ, inverse_prop x) :=
sorry

end NUMINAMATH_CALUDE_inverse_proposition_l2802_280248


namespace NUMINAMATH_CALUDE_negation_of_square_positive_equals_zero_l2802_280200

theorem negation_of_square_positive_equals_zero (m : ℝ) :
  ¬(m > 0 ∧ m^2 = 0) ↔ (m ≤ 0 → m^2 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_square_positive_equals_zero_l2802_280200


namespace NUMINAMATH_CALUDE_math_game_result_l2802_280249

theorem math_game_result (a : ℚ) : 
  (1/2 : ℚ) * (-(- a) - 2) = -1/2 * a - 1 := by
  sorry

end NUMINAMATH_CALUDE_math_game_result_l2802_280249


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_185_l2802_280298

theorem modular_inverse_of_3_mod_185 :
  ∃ x : ℕ, x < 185 ∧ (3 * x) % 185 = 1 :=
by
  use 62
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_185_l2802_280298
