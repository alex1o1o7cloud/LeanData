import Mathlib

namespace NUMINAMATH_CALUDE_total_cats_is_twenty_l3065_306582

-- Define the number of cats for each person
def jamie_persian : Nat := 4
def jamie_maine_coon : Nat := 2
def gordon_persian : Nat := jamie_persian / 2
def gordon_maine_coon : Nat := jamie_maine_coon + 1
def hawkeye_persian : Nat := 0
def hawkeye_maine_coon : Nat := gordon_maine_coon - 1
def natasha_persian : Nat := 3
def natasha_maine_coon : Nat := 4

-- Define the total number of cats
def total_cats : Nat :=
  jamie_persian + jamie_maine_coon +
  gordon_persian + gordon_maine_coon +
  hawkeye_persian + hawkeye_maine_coon +
  natasha_persian + natasha_maine_coon

-- Theorem to prove
theorem total_cats_is_twenty : total_cats = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_is_twenty_l3065_306582


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l3065_306568

/-- Given a total of 68 students in eighth grade with 28 girls, 
    the ratio of boys to girls is 10:7. -/
theorem boys_to_girls_ratio : 
  let total_students : ℕ := 68
  let girls : ℕ := 28
  let boys : ℕ := total_students - girls
  ∃ (a b : ℕ), a = 10 ∧ b = 7 ∧ boys * b = girls * a :=
by sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l3065_306568


namespace NUMINAMATH_CALUDE_complex_coordinates_l3065_306531

theorem complex_coordinates (z : ℂ) : z = (2 + Complex.I) / Complex.I → 
  Complex.re z = 1 ∧ Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinates_l3065_306531


namespace NUMINAMATH_CALUDE_x_value_l3065_306599

theorem x_value : ∃ x : ℚ, (3 * x) / 7 = 12 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3065_306599


namespace NUMINAMATH_CALUDE_gcd_factorial_8_9_l3065_306511

theorem gcd_factorial_8_9 : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_8_9_l3065_306511


namespace NUMINAMATH_CALUDE_circle_radii_relation_l3065_306501

/-- Given three circles with centers A, B, C, touching each other and a line l,
    with radii a, b, and c respectively, prove that 1/√c = 1/√a + 1/√b. -/
theorem circle_radii_relation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / Real.sqrt c = 1 / Real.sqrt a + 1 / Real.sqrt b := by
sorry

end NUMINAMATH_CALUDE_circle_radii_relation_l3065_306501


namespace NUMINAMATH_CALUDE_inequality_proof_l3065_306579

theorem inequality_proof (a b c d e f : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e) (h6 : e ≤ f) :
  (a * f + b * e + c * d) * (a * f + b * d + c * e) ≤ (a + b^2 + c^3) * (d + e^2 + f^3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3065_306579


namespace NUMINAMATH_CALUDE_negation_equivalence_l3065_306549

variable (a : ℝ)

theorem negation_equivalence :
  (¬ ∀ x : ℝ, (x - a)^2 + 2 > 0) ↔ (∃ x : ℝ, (x - a)^2 + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3065_306549


namespace NUMINAMATH_CALUDE_focus_of_our_parabola_l3065_306507

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- A parabola defined by its equation -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- The focus of a parabola given its equation -/
def focus_of_parabola (p : Parabola) : Focus := sorry

/-- The parabola x^2 + y = 0 -/
def our_parabola : Parabola :=
  { equation := fun x y => x^2 + y = 0 }

theorem focus_of_our_parabola :
  focus_of_parabola our_parabola = ⟨0, -1/4⟩ := by sorry

end NUMINAMATH_CALUDE_focus_of_our_parabola_l3065_306507


namespace NUMINAMATH_CALUDE_total_boxes_needed_l3065_306523

-- Define the amounts of wooden sticks and box capacities
def taehyung_total : ℚ := 21 / 11
def taehyung_per_box : ℚ := 7 / 11
def hoseok_total : ℚ := 8 / 17
def hoseok_per_box : ℚ := 2 / 17

-- Define the function to calculate the number of boxes needed
def boxes_needed (total : ℚ) (per_box : ℚ) : ℕ :=
  (total / per_box).ceil.toNat

-- Theorem statement
theorem total_boxes_needed :
  boxes_needed taehyung_total taehyung_per_box +
  boxes_needed hoseok_total hoseok_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_boxes_needed_l3065_306523


namespace NUMINAMATH_CALUDE_bakers_purchase_problem_l3065_306580

/-- A baker's purchase problem -/
theorem bakers_purchase_problem 
  (total_cost : ℕ)
  (flour_cost : ℕ)
  (egg_cost : ℕ)
  (egg_quantity : ℕ)
  (milk_cost : ℕ)
  (milk_quantity : ℕ)
  (soda_cost : ℕ)
  (soda_quantity : ℕ)
  (h1 : total_cost = 80)
  (h2 : flour_cost = 3)
  (h3 : egg_cost = 10)
  (h4 : egg_quantity = 3)
  (h5 : milk_cost = 5)
  (h6 : milk_quantity = 7)
  (h7 : soda_cost = 3)
  (h8 : soda_quantity = 2) :
  ∃ (flour_quantity : ℕ), 
    flour_quantity * flour_cost + 
    egg_quantity * egg_cost + 
    milk_quantity * milk_cost + 
    soda_quantity * soda_cost = total_cost ∧ 
    flour_quantity = 3 :=
by sorry

end NUMINAMATH_CALUDE_bakers_purchase_problem_l3065_306580


namespace NUMINAMATH_CALUDE_right_triangle_area_l3065_306520

theorem right_triangle_area (h : ℝ) (θ : ℝ) (area : ℝ) : 
  h = 12 →  -- hypotenuse is 12 inches
  θ = 30 * π / 180 →  -- one angle is 30 degrees (converted to radians)
  area = 18 * Real.sqrt 3 →  -- area is 18√3 square inches
  area = (h * h * Real.sin θ * Real.cos θ) / 2 :=
by sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_right_triangle_area_l3065_306520


namespace NUMINAMATH_CALUDE_waste_paper_collection_l3065_306567

/-- Proves that given the conditions of the waste paper collection problem,
    Vitya collected 15 kg and Vova collected 12 kg. -/
theorem waste_paper_collection :
  ∀ (v w : ℕ),
  v + w = 27 →
  5 * v + 3 * w = 111 →
  v = 15 ∧ w = 12 := by
sorry

end NUMINAMATH_CALUDE_waste_paper_collection_l3065_306567


namespace NUMINAMATH_CALUDE_ellipse_intersection_midpoint_l3065_306573

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  y = x + 2

-- Theorem statement
theorem ellipse_intersection_midpoint :
  -- Given conditions
  let f1 : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
  let f2 : ℝ × ℝ := (2 * Real.sqrt 2, 0)
  let major_axis_length : ℝ := 6

  -- Prove that
  -- 1. The standard equation of ellipse C is x²/9 + y² = 1
  (∀ x y : ℝ, ellipse_C x y ↔ x^2 / 9 + y^2 = 1) ∧
  -- 2. The midpoint of intersection points has coordinates (-9/5, 1/5)
  (∃ x1 y1 x2 y2 : ℝ,
    ellipse_C x1 y1 ∧ ellipse_C x2 y2 ∧
    line x1 y1 ∧ line x2 y2 ∧
    x1 ≠ x2 ∧
    (x1 + x2) / 2 = -9/5 ∧
    (y1 + y2) / 2 = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_midpoint_l3065_306573


namespace NUMINAMATH_CALUDE_bus_stop_time_l3065_306514

/-- Calculates the stop time of a bus given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) :
  speed_without_stops = 54 →
  speed_with_stops = 45 →
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 10 := by
  sorry

#check bus_stop_time

end NUMINAMATH_CALUDE_bus_stop_time_l3065_306514


namespace NUMINAMATH_CALUDE_certain_number_multiplication_l3065_306572

theorem certain_number_multiplication : ∃ x : ℤ, (x - 7 = 9) ∧ (x * 3 = 48) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_multiplication_l3065_306572


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l3065_306565

theorem binomial_coefficient_congruence (p n : ℕ) (hp : Prime p) :
  (Nat.choose (n * p) n) ≡ n [MOD p^2] := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l3065_306565


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3065_306576

theorem fraction_evaluation : (16 + 8) / (4 - 2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3065_306576


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l3065_306560

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane
  (m n : Line) (α : Plane)
  (h1 : perpendicular m n)
  (h2 : perpendicularLP n α)
  (h3 : ¬ intersects m α) :
  parallel m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l3065_306560


namespace NUMINAMATH_CALUDE_bart_firewood_needs_l3065_306587

/-- The number of pieces of firewood obtained from one tree -/
def pieces_per_tree : ℕ := 75

/-- The number of logs burned per day -/
def logs_per_day : ℕ := 5

/-- The number of days from November 1 through February 28 -/
def total_days : ℕ := 120

/-- The number of trees Bart needs to cut down -/
def trees_needed : ℕ := 8

theorem bart_firewood_needs :
  trees_needed = (total_days * logs_per_day + pieces_per_tree - 1) / pieces_per_tree :=
by sorry

end NUMINAMATH_CALUDE_bart_firewood_needs_l3065_306587


namespace NUMINAMATH_CALUDE_special_sequence_properties_l3065_306504

/-- A sequence and its partial sums satisfying certain conditions -/
structure SpecialSequence where
  q : ℝ
  a : ℕ → ℝ
  S : ℕ → ℝ
  h1 : q * (q - 1) ≠ 0
  h2 : ∀ n : ℕ, (1 - q) * S n + q^n = 1
  h3 : S 3 - S 9 = S 9 - S 6

/-- The main theorem about the special sequence -/
theorem special_sequence_properties (seq : SpecialSequence) :
  (∀ n : ℕ, seq.a n = seq.q^(n - 1)) ∧
  (seq.a 2 - seq.a 8 = seq.a 8 - seq.a 5) := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_properties_l3065_306504


namespace NUMINAMATH_CALUDE_even_function_inequality_l3065_306563

/-- An even function from ℝ to ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem even_function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (hf_even : EvenFunction f)
  (hf_deriv : ∀ x, HasDerivAt f (f' x) x)
  (h_ineq : ∀ x, 2 * f x + x * f' x < 2) :
  ∀ x, x^2 * f x - f 1 < x^2 - 1 ↔ |x| > 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3065_306563


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l3065_306561

theorem cube_sum_of_roots (u v w : ℝ) : 
  (u - Real.rpow 17 (1/3 : ℝ)) * (u - Real.rpow 67 (1/3 : ℝ)) * (u - Real.rpow 127 (1/3 : ℝ)) = 1/4 →
  (v - Real.rpow 17 (1/3 : ℝ)) * (v - Real.rpow 67 (1/3 : ℝ)) * (v - Real.rpow 127 (1/3 : ℝ)) = 1/4 →
  (w - Real.rpow 17 (1/3 : ℝ)) * (w - Real.rpow 67 (1/3 : ℝ)) * (w - Real.rpow 127 (1/3 : ℝ)) = 1/4 →
  u ≠ v → u ≠ w → v ≠ w →
  u^3 + v^3 + w^3 = 211.75 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l3065_306561


namespace NUMINAMATH_CALUDE_original_price_proof_l3065_306540

/-- Represents the discount rate as a fraction -/
def discount_rate : ℚ := 1 / 10

/-- Calculates the original price before discounts -/
def original_price (final_price : ℚ) : ℚ :=
  final_price / (1 - discount_rate)

/-- The final price after discounts -/
def final_price : ℚ := 230

theorem original_price_proof :
  ∃ (price : ℕ), price ≥ 256 ∧ price < 257 ∧ 
  (original_price final_price).num / (original_price final_price).den = price / 1 := by
  sorry

#eval (original_price final_price).num / (original_price final_price).den

end NUMINAMATH_CALUDE_original_price_proof_l3065_306540


namespace NUMINAMATH_CALUDE_square_cut_rectangle_perimeter_l3065_306509

/-- Given a square with perimeter 20 cm cut into two rectangles, 
    where one rectangle has perimeter 16 cm, 
    prove that the other rectangle has perimeter 14 cm. -/
theorem square_cut_rectangle_perimeter :
  ∀ (square_perimeter : ℝ) (rectangle1_perimeter : ℝ),
    square_perimeter = 20 →
    rectangle1_perimeter = 16 →
    ∃ (rectangle2_perimeter : ℝ),
      rectangle2_perimeter = 14 ∧
      rectangle1_perimeter + rectangle2_perimeter = square_perimeter + 10 :=
by sorry

end NUMINAMATH_CALUDE_square_cut_rectangle_perimeter_l3065_306509


namespace NUMINAMATH_CALUDE_smallest_positive_d_smallest_d_is_zero_l3065_306517

theorem smallest_positive_d (d : ℝ) (hd : d > 0) :
  ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + d * (x - y)^2 ≥ (x + y) / 2 := by
  sorry

/-- The smallest positive real number d that satisfies the inequality for all nonnegative x and y is 0 -/
theorem smallest_d_is_zero :
  ∀ ε > 0, ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ Real.sqrt (x * y) + ε * (x - y)^2 < (x + y) / 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_d_smallest_d_is_zero_l3065_306517


namespace NUMINAMATH_CALUDE_chocolate_milk_remaining_l3065_306534

/-- The amount of chocolate milk remaining after drinking some on two consecutive days. -/
theorem chocolate_milk_remaining (initial : ℝ) (day1 : ℝ) (day2 : ℝ) (h1 : initial = 1.6) (h2 : day1 = 0.8) (h3 : day2 = 0.3) :
  initial - day1 - day2 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_remaining_l3065_306534


namespace NUMINAMATH_CALUDE_interior_perimeter_is_20_l3065_306525

/-- Represents a rectangular picture frame -/
structure Frame where
  outer_width : ℝ
  outer_height : ℝ
  border_width : ℝ
  frame_area : ℝ

/-- Calculates the sum of the lengths of the four interior edges of a frame -/
def interior_perimeter (f : Frame) : ℝ :=
  2 * ((f.outer_width - 2 * f.border_width) + (f.outer_height - 2 * f.border_width))

/-- Theorem stating that for a frame with given dimensions, the interior perimeter is 20 inches -/
theorem interior_perimeter_is_20 (f : Frame) 
  (h_outer_width : f.outer_width = 8)
  (h_outer_height : f.outer_height = 10)
  (h_border_width : f.border_width = 2)
  (h_frame_area : f.frame_area = 52) :
  interior_perimeter f = 20 := by
  sorry

#check interior_perimeter_is_20

end NUMINAMATH_CALUDE_interior_perimeter_is_20_l3065_306525


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_hyperbola_eccentricity_is_sqrt_5_l3065_306564

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let P : ℝ × ℝ := sorry
  let F₁ : ℝ × ℝ := sorry
  let F₂ : ℝ × ℝ := sorry
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let circle := fun (x y : ℝ) ↦ x^2 + y^2 = a^2 + b^2
  let distance := fun (p q : ℝ × ℝ) ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  have h1 : hyperbola P.1 P.2 := sorry
  have h2 : circle P.1 P.2 := sorry
  have h3 : P.1 ≥ 0 ∧ P.2 ≥ 0 := sorry  -- P is in the first quadrant
  have h4 : distance P F₁ = 2 * distance P F₂ := sorry
  Real.sqrt 5

theorem hyperbola_eccentricity_is_sqrt_5 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola_eccentricity a b ha hb = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_hyperbola_eccentricity_is_sqrt_5_l3065_306564


namespace NUMINAMATH_CALUDE_oranges_sold_l3065_306529

def total_bags : ℕ := 10
def oranges_per_bag : ℕ := 30
def rotten_oranges : ℕ := 50
def oranges_for_juice : ℕ := 30

theorem oranges_sold : 
  (total_bags * oranges_per_bag - rotten_oranges - oranges_for_juice) = 220 := by
  sorry

end NUMINAMATH_CALUDE_oranges_sold_l3065_306529


namespace NUMINAMATH_CALUDE_water_volume_in_cone_l3065_306553

/-- 
Theorem: For a cone filled with water up to 2/3 of its height, 
the volume of water is 8/27 of the total volume of the cone.
-/
theorem water_volume_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  (1/3 * π * (2/3 * r)^2 * (2/3 * h)) / (1/3 * π * r^2 * h) = 8/27 := by
  sorry


end NUMINAMATH_CALUDE_water_volume_in_cone_l3065_306553


namespace NUMINAMATH_CALUDE_day_after_tomorrow_l3065_306586

/-- Represents days of the week -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the next day -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

/-- Returns the previous day -/
def prevDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Saturday
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday

theorem day_after_tomorrow (today : Day) :
  (nextDay (nextDay today) = Day.Saturday) → (today = Day.Thursday) →
  (prevDay (nextDay (nextDay today)) = Day.Friday) :=
by
  sorry


end NUMINAMATH_CALUDE_day_after_tomorrow_l3065_306586


namespace NUMINAMATH_CALUDE_inequality_max_m_l3065_306562

theorem inequality_max_m : 
  ∀ (m : ℝ), 
  (∀ (a b : ℝ), a > 0 → b > 0 → (2/a + 1/b ≥ m/(2*a+b))) ↔ m ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_inequality_max_m_l3065_306562


namespace NUMINAMATH_CALUDE_imaginary_sum_zero_l3065_306571

theorem imaginary_sum_zero (i : ℂ) (h : i^2 = -1) :
  i^15324 + i^15325 + i^15326 + i^15327 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_zero_l3065_306571


namespace NUMINAMATH_CALUDE_parabola_equation_l3065_306524

/-- Given a parabola with directrix x = -7, its standard equation is y^2 = 28x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ x = -7) →  -- directrix equation
  (∃ a b c : ℝ, ∀ x y, p (x, y) ↔ y^2 = 28*x + b*y + c) -- standard form of parabola
  :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3065_306524


namespace NUMINAMATH_CALUDE_find_divisor_l3065_306584

theorem find_divisor (D : ℕ) (x : ℕ) : 
  (∃ (x : ℕ), x ≤ 11 ∧ (2000 - x) % D = 0) → 
  (2000 - x = 1989) →
  D = 11 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l3065_306584


namespace NUMINAMATH_CALUDE_shirt_cost_l3065_306548

theorem shirt_cost (J S K : ℚ) 
  (eq1 : 3 * J + 2 * S + K = 110)
  (eq2 : 2 * J + 3 * S + 2 * K = 176)
  (eq3 : 4 * J + S + 3 * K = 254) :
  S = 5.6 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l3065_306548


namespace NUMINAMATH_CALUDE_gcf_of_3150_and_9800_l3065_306559

theorem gcf_of_3150_and_9800 : Nat.gcd 3150 9800 = 350 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_3150_and_9800_l3065_306559


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3065_306515

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 5 ∧ b > 5 ∧
    n = 1 * a + 5 ∧
    n = 5 * b + 1

theorem smallest_dual_base_representation :
  (∀ m : ℕ, m < 31 → ¬ is_valid_representation m) ∧
  is_valid_representation 31 :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3065_306515


namespace NUMINAMATH_CALUDE_additional_pecks_needed_to_fill_barrel_l3065_306506

-- Define the relationships
def peck_to_bushel : ℚ := 1/4
def bushel_to_barrel : ℚ := 1/9

-- Define the number of pecks already picked
def pecks_picked : ℕ := 1

-- Theorem statement
theorem additional_pecks_needed_to_fill_barrel : 
  ∀ (pecks_in_barrel : ℕ), 
    pecks_in_barrel = (1 / peck_to_bushel : ℚ) * (1 / bushel_to_barrel : ℚ) → 
    pecks_in_barrel - pecks_picked = 35 := by
  sorry

end NUMINAMATH_CALUDE_additional_pecks_needed_to_fill_barrel_l3065_306506


namespace NUMINAMATH_CALUDE_cross_product_example_l3065_306596

/-- The cross product of two 3D vectors -/
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1 - u.1 * v.2.2,
   u.1 * v.2.1 - u.2.1 * v.1)

theorem cross_product_example : 
  cross_product (3, -4, 7) (2, 5, -3) = (-23, 23, 23) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_example_l3065_306596


namespace NUMINAMATH_CALUDE_ashwin_rental_hours_verify_solution_l3065_306589

/-- Calculates the total rental hours given the rental conditions and total amount paid --/
def rental_hours (first_hour_cost : ℕ) (additional_hour_cost : ℕ) (total_paid : ℕ) : ℕ :=
  let additional_hours := (total_paid - first_hour_cost) / additional_hour_cost
  1 + additional_hours

/-- Proves that Ashwin rented the tool for 11 hours given the specified conditions --/
theorem ashwin_rental_hours :
  rental_hours 25 10 125 = 11 := by
  sorry

/-- Verifies the solution satisfies the original problem conditions --/
theorem verify_solution :
  25 + 10 * (rental_hours 25 10 125 - 1) = 125 := by
  sorry

end NUMINAMATH_CALUDE_ashwin_rental_hours_verify_solution_l3065_306589


namespace NUMINAMATH_CALUDE_pumpkin_count_l3065_306543

/-- The total number of pumpkins grown by Sandy, Mike, Maria, and Sam -/
def total_pumpkins (sandy mike maria sam : ℕ) : ℕ := sandy + mike + maria + sam

/-- Theorem stating that the total number of pumpkins is 157 -/
theorem pumpkin_count : total_pumpkins 51 23 37 46 = 157 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_count_l3065_306543


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_l3065_306585

theorem least_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  3 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) → 3 ∣ m → 7 ∣ m → 11 ∣ m → n ≤ m) ∧
  n = 1155 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_l3065_306585


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3065_306575

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 12) (h2 : x ≠ -4) :
  (7 * x - 5) / (x^2 - 8*x - 48) = (79/16) / (x - 12) + (33/16) / (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3065_306575


namespace NUMINAMATH_CALUDE_factorization_of_M_l3065_306519

theorem factorization_of_M (a b c d : ℝ) :
  ((a - c)^2 + (b - d)^2) * (a^2 + b^2) - (a * d - b * c)^2 = (a * c + b * d - a^2 - b^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_M_l3065_306519


namespace NUMINAMATH_CALUDE_george_dimes_count_l3065_306577

/-- Prove the number of dimes in George's collection --/
theorem george_dimes_count :
  let total_coins : ℕ := 28
  let total_value : ℚ := 260/100
  let nickel_count : ℕ := 4
  let nickel_value : ℚ := 5/100
  let dime_value : ℚ := 10/100
  ∃ dime_count : ℕ,
    dime_count = 24 ∧
    dime_count + nickel_count = total_coins ∧
    dime_count * dime_value + nickel_count * nickel_value = total_value :=
by sorry

end NUMINAMATH_CALUDE_george_dimes_count_l3065_306577


namespace NUMINAMATH_CALUDE_count_divisible_by_2_3_or_5_count_divisible_by_2_3_or_5_is_74_l3065_306588

theorem count_divisible_by_2_3_or_5 : ℕ :=
  let n := 100
  let A₂ := n / 2
  let A₃ := n / 3
  let A₅ := n / 5
  let A₂₃ := n / 6
  let A₂₅ := n / 10
  let A₃₅ := n / 15
  let A₂₃₅ := n / 30
  A₂ + A₃ + A₅ - A₂₃ - A₂₅ - A₃₅ + A₂₃₅

theorem count_divisible_by_2_3_or_5_is_74 : 
  count_divisible_by_2_3_or_5 = 74 := by sorry

end NUMINAMATH_CALUDE_count_divisible_by_2_3_or_5_count_divisible_by_2_3_or_5_is_74_l3065_306588


namespace NUMINAMATH_CALUDE_function_properties_l3065_306522

-- Define the function f
def f (a b c x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

-- State the theorem
theorem function_properties (a b c : ℝ) :
  (∀ x < 0, ∀ y < x, f a b c x < f a b c y) →  -- f is decreasing on (-∞, 0)
  (∀ x ∈ Set.Ioo 0 1, ∀ y ∈ Set.Ioo 0 1, x < y → f a b c x < f a b c y) →  -- f is increasing on (0, 1)
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b c x = 0 ∧ f a b c y = 0 ∧ f a b c z = 0) →  -- f has three real roots
  f a b c 1 = 0 →  -- 1 is a root of f
  b = 0 ∧ f a b c 2 > -5/2 ∧ 3/2 < a ∧ a < 2 * Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3065_306522


namespace NUMINAMATH_CALUDE_time_to_see_again_is_48_l3065_306558

-- Define the parameters of the problem
def path_distance : ℝ := 300
def building_diameter : ℝ := 150
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def initial_distance : ℝ := 300

-- Define the function to calculate the time until they can see each other again
def time_to_see_again (pd : ℝ) (bd : ℝ) (ks : ℝ) (js : ℝ) (id : ℝ) : ℝ :=
  -- The actual calculation would go here, but we'll use sorry to skip the proof
  sorry

-- State the theorem
theorem time_to_see_again_is_48 :
  time_to_see_again path_distance building_diameter kenny_speed jenny_speed initial_distance = 48 := by
  sorry

end NUMINAMATH_CALUDE_time_to_see_again_is_48_l3065_306558


namespace NUMINAMATH_CALUDE_smallest_divisible_by_three_l3065_306594

theorem smallest_divisible_by_three :
  ∃ (B : ℕ), B < 10 ∧ 
    (∀ (k : ℕ), k < B → ¬(800000 + 100000 * k + 4635) % 3 = 0) ∧
    (800000 + 100000 * B + 4635) % 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_three_l3065_306594


namespace NUMINAMATH_CALUDE_vasya_reads_entire_book_l3065_306521

theorem vasya_reads_entire_book :
  let first_day : ℚ := 1/2
  let second_day : ℚ := (1/3) * (1 - first_day)
  let first_two_days : ℚ := first_day + second_day
  let third_day : ℚ := (1/2) * first_two_days
  first_day + second_day + third_day = 1 := by sorry

end NUMINAMATH_CALUDE_vasya_reads_entire_book_l3065_306521


namespace NUMINAMATH_CALUDE_linear_equation_exponent_l3065_306550

theorem linear_equation_exponent (n : ℕ) : 
  (∀ x, ∃ a b, x^(2*n - 5) - 2 = a*x + b) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_l3065_306550


namespace NUMINAMATH_CALUDE_square_properties_l3065_306541

/-- Given a square and a rectangle with the same perimeter, where the rectangle
    has sides of 10 cm and 7 cm, this theorem proves the side length and area of the square. -/
theorem square_properties (square_perimeter rectangle_perimeter : ℝ)
                          (rectangle_side1 rectangle_side2 : ℝ)
                          (h1 : square_perimeter = rectangle_perimeter)
                          (h2 : rectangle_side1 = 10)
                          (h3 : rectangle_side2 = 7)
                          (h4 : rectangle_perimeter = 2 * (rectangle_side1 + rectangle_side2)) :
  ∃ (square_side : ℝ),
    square_side = 8.5 ∧
    square_perimeter = 4 * square_side ∧
    square_side ^ 2 = 72.25 := by
  sorry

end NUMINAMATH_CALUDE_square_properties_l3065_306541


namespace NUMINAMATH_CALUDE_difference_of_reciprocals_l3065_306537

theorem difference_of_reciprocals (p q : ℚ) 
  (hp : 3 / p = 6) 
  (hq : 3 / q = 15) : 
  p - q = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_difference_of_reciprocals_l3065_306537


namespace NUMINAMATH_CALUDE_can_lids_per_box_l3065_306554

theorem can_lids_per_box 
  (num_boxes : ℕ) 
  (initial_lids : ℕ) 
  (final_total_lids : ℕ) 
  (h1 : num_boxes = 3) 
  (h2 : initial_lids = 14) 
  (h3 : final_total_lids = 53) :
  (final_total_lids - initial_lids) / num_boxes = 13 := by
  sorry

#check can_lids_per_box

end NUMINAMATH_CALUDE_can_lids_per_box_l3065_306554


namespace NUMINAMATH_CALUDE_x_difference_l3065_306502

theorem x_difference (x₁ x₂ : ℝ) : 
  ((x₁ + 3)^2 / (3*x₁ + 65) = 2) →
  ((x₂ + 3)^2 / (3*x₂ + 65) = 2) →
  x₁ ≠ x₂ →
  |x₁ - x₂| = 22 := by
sorry

end NUMINAMATH_CALUDE_x_difference_l3065_306502


namespace NUMINAMATH_CALUDE_students_without_pens_l3065_306547

theorem students_without_pens (total students_with_blue students_with_red students_with_both : ℕ) 
  (h1 : total = 40)
  (h2 : students_with_blue = 18)
  (h3 : students_with_red = 26)
  (h4 : students_with_both = 10) :
  total - (students_with_blue + students_with_red - students_with_both) = 6 :=
by sorry

end NUMINAMATH_CALUDE_students_without_pens_l3065_306547


namespace NUMINAMATH_CALUDE_prob_ace_king_queen_value_l3065_306526

/-- The probability of drawing an Ace, then a King, then a Queen from a standard deck of 52 cards without replacement -/
def prob_ace_king_queen : ℚ :=
  let total_cards : ℕ := 52
  let num_aces : ℕ := 4
  let num_kings : ℕ := 4
  let num_queens : ℕ := 4
  (num_aces : ℚ) / total_cards *
  (num_kings : ℚ) / (total_cards - 1) *
  (num_queens : ℚ) / (total_cards - 2)

theorem prob_ace_king_queen_value : prob_ace_king_queen = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_king_queen_value_l3065_306526


namespace NUMINAMATH_CALUDE_partner_A_investment_l3065_306574

/-- Calculates the investment of partner A in a business partnership --/
theorem partner_A_investment
  (b_investment : ℕ)
  (c_investment : ℕ)
  (total_profit : ℕ)
  (a_profit_share : ℕ)
  (h1 : b_investment = 4200)
  (h2 : c_investment = 10500)
  (h3 : total_profit = 12200)
  (h4 : a_profit_share = 3660) :
  ∃ a_investment : ℕ,
    a_investment = 6725 ∧
    a_investment * total_profit = a_profit_share * (a_investment + b_investment + c_investment) :=
by
  sorry


end NUMINAMATH_CALUDE_partner_A_investment_l3065_306574


namespace NUMINAMATH_CALUDE_investment_growth_l3065_306532

theorem investment_growth (initial_investment : ℝ) (first_year_loss_rate : ℝ) (second_year_gain_rate : ℝ) :
  initial_investment = 150 →
  first_year_loss_rate = 0.1 →
  second_year_gain_rate = 0.25 →
  let first_year_amount := initial_investment * (1 - first_year_loss_rate)
  let final_amount := first_year_amount * (1 + second_year_gain_rate)
  let overall_gain_rate := (final_amount - initial_investment) / initial_investment
  overall_gain_rate = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l3065_306532


namespace NUMINAMATH_CALUDE_jane_cans_count_l3065_306595

theorem jane_cans_count (total_seeds : ℝ) (seeds_per_can : ℕ) (h1 : total_seeds = 54.0) (h2 : seeds_per_can = 6) :
  (total_seeds / seeds_per_can : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_jane_cans_count_l3065_306595


namespace NUMINAMATH_CALUDE_inscribed_circle_theorem_l3065_306538

def equilateral_triangle_with_inscribed_circle 
  (radius : ℝ) (height : ℝ) (x : ℝ) : Prop :=
  radius = 3/16 ∧ 
  height = 3 * radius ∧ 
  x = height - 1/2

theorem inscribed_circle_theorem :
  ∀ (radius height x : ℝ),
    equilateral_triangle_with_inscribed_circle radius height x →
    x = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_theorem_l3065_306538


namespace NUMINAMATH_CALUDE_children_got_on_bus_l3065_306557

/-- Proves that the number of children who got on the bus at the bus stop is 14 -/
theorem children_got_on_bus (initial_children : ℕ) (final_children : ℕ) 
  (h1 : initial_children = 64) (h2 : final_children = 78) : 
  final_children - initial_children = 14 := by
  sorry

end NUMINAMATH_CALUDE_children_got_on_bus_l3065_306557


namespace NUMINAMATH_CALUDE_negative_a_sixth_divided_by_a_third_l3065_306593

theorem negative_a_sixth_divided_by_a_third (a : ℝ) : (-a)^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_sixth_divided_by_a_third_l3065_306593


namespace NUMINAMATH_CALUDE_opposite_of_three_l3065_306590

theorem opposite_of_three : -(3 : ℝ) = -3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_three_l3065_306590


namespace NUMINAMATH_CALUDE_sequence_general_term_l3065_306592

/-- A sequence satisfying the given recurrence relation -/
def Sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n + 2 * a (n + 1) = 7 * 3^(n - 1)) ∧ a 1 = 1

/-- Theorem stating that the sequence has the general term a_n = 3^(n-1) -/
theorem sequence_general_term (a : ℕ → ℝ) (h : Sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = 3^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3065_306592


namespace NUMINAMATH_CALUDE_percent_equality_l3065_306555

theorem percent_equality (x : ℝ) (h : (0.3 * (0.2 * x)) = 24) :
  (0.2 * (0.3 * x)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l3065_306555


namespace NUMINAMATH_CALUDE_complex_power_sum_l3065_306542

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^1500 + 1/(z^1500) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3065_306542


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3065_306516

theorem partial_fraction_decomposition (x : ℝ) (A B : ℚ) : 
  (5 * x - 3) / ((x - 3) * (x + 6)) = A / (x - 3) + B / (x + 6) ↔ 
  A = 4/3 ∧ B = 11/3 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3065_306516


namespace NUMINAMATH_CALUDE_ball_color_distribution_l3065_306535

theorem ball_color_distribution (x y z : ℕ) : 
  x + y + z = 20 →
  x > 0 ∧ y > 0 ∧ z > 0 →
  (z : ℚ) / 20 - (2 * x : ℚ) / (2 * x + y + z) = 1 / 5 →
  x = 5 ∧ y = 3 ∧ z = 12 := by
sorry

end NUMINAMATH_CALUDE_ball_color_distribution_l3065_306535


namespace NUMINAMATH_CALUDE_complex_modulus_l3065_306569

theorem complex_modulus (z : ℂ) :
  (((2 : ℂ) + 4 * I) / z = 1 + I) → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3065_306569


namespace NUMINAMATH_CALUDE_shorter_tank_radius_l3065_306597

/-- Given two cylindrical tanks with equal volume, where one tank's height is four times the other
    and the taller tank has a radius of 10 units, the radius of the shorter tank is 20 units. -/
theorem shorter_tank_radius (h : ℝ) (h_pos : h > 0) : 
  π * (10 ^ 2) * (4 * h) = π * (20 ^ 2) * h := by sorry

end NUMINAMATH_CALUDE_shorter_tank_radius_l3065_306597


namespace NUMINAMATH_CALUDE_problem_solution_l3065_306527

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^5 + 2*y^3) / 8 = 46.375 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3065_306527


namespace NUMINAMATH_CALUDE_mobile_phone_price_decrease_l3065_306551

theorem mobile_phone_price_decrease (current_price : ℝ) (yearly_decrease_rate : ℝ) (years : ℕ) : 
  current_price = 1000 ∧ yearly_decrease_rate = 0.2 ∧ years = 2 →
  current_price = (1562.5 : ℝ) * (1 - yearly_decrease_rate) ^ years := by
sorry

end NUMINAMATH_CALUDE_mobile_phone_price_decrease_l3065_306551


namespace NUMINAMATH_CALUDE_original_average_problem_l3065_306536

theorem original_average_problem (n : ℕ) (original_avg new_avg added : ℝ) : 
  n = 15 → 
  new_avg = 51 → 
  added = 11 → 
  (n : ℝ) * new_avg = (n : ℝ) * (original_avg + added) → 
  original_avg = 40 := by
sorry

end NUMINAMATH_CALUDE_original_average_problem_l3065_306536


namespace NUMINAMATH_CALUDE_garden_length_l3065_306530

/-- Given a rectangular garden with area 120 m², if reducing its length by 2m results in a square,
    then the original length of the garden is 12 meters. -/
theorem garden_length (length width : ℝ) : 
  length * width = 120 →
  (length - 2) * (length - 2) = width * (length - 2) →
  length = 12 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l3065_306530


namespace NUMINAMATH_CALUDE_circumscribed_trapezoid_inequality_l3065_306583

/-- A trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- Radius of the inscribed circle -/
  R : ℝ
  /-- Length of one base of the trapezoid -/
  a : ℝ
  /-- Length of the other base of the trapezoid -/
  b : ℝ
  /-- The trapezoid is circumscribed around the circle -/
  circumscribed : True

/-- For a trapezoid circumscribed around a circle with radius R and bases a and b, ab ≥ 4R^2 -/
theorem circumscribed_trapezoid_inequality (t : CircumscribedTrapezoid) : t.a * t.b ≥ 4 * t.R^2 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_trapezoid_inequality_l3065_306583


namespace NUMINAMATH_CALUDE_basketball_weight_is_20_l3065_306578

/-- The weight of one basketball in pounds -/
def basketball_weight : ℝ := 20

/-- The weight of one bicycle in pounds -/
def bicycle_weight : ℝ := 40

/-- Theorem stating that one basketball weighs 20 pounds given the conditions -/
theorem basketball_weight_is_20 :
  (8 * basketball_weight = 4 * bicycle_weight) →
  (3 * bicycle_weight = 120) →
  basketball_weight = 20 := by
sorry

end NUMINAMATH_CALUDE_basketball_weight_is_20_l3065_306578


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3065_306598

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.length)

/-- Theorem: A rectangle with perimeter 150 cm and length 15 cm greater than width
    has width 30 cm and length 45 cm -/
theorem rectangle_dimensions :
  ∃ (r : Rectangle),
    perimeter r = 150 ∧
    r.length = r.width + 15 ∧
    r.width = 30 ∧
    r.length = 45 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3065_306598


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l3065_306581

/-- Given a line segment with midpoint (-1, 3) and one endpoint (2, -4),
    prove that the other endpoint is (-4, 10). -/
theorem other_endpoint_of_line_segment
  (midpoint : ℝ × ℝ)
  (endpoint1 : ℝ × ℝ)
  (h_midpoint : midpoint = (-1, 3))
  (h_endpoint1 : endpoint1 = (2, -4)) :
  ∃ (endpoint2 : ℝ × ℝ),
    endpoint2 = (-4, 10) ∧
    midpoint = (
      (endpoint1.1 + endpoint2.1) / 2,
      (endpoint1.2 + endpoint2.2) / 2
    ) :=
by sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l3065_306581


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3065_306528

/-- Given the ages of Tony and Belinda, prove that their age ratio is 5/2 -/
theorem age_ratio_proof (tony_age belinda_age : ℕ) : 
  tony_age = 16 →
  belinda_age = 40 →
  tony_age + belinda_age = 56 →
  ∃ (k : ℕ), belinda_age = k * tony_age + 8 →
  (belinda_age : ℚ) / (tony_age : ℚ) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3065_306528


namespace NUMINAMATH_CALUDE_eel_species_count_l3065_306552

/-- Given the number of species identified in an aquarium, prove the number of eel species. -/
theorem eel_species_count (total : ℕ) (sharks : ℕ) (whales : ℕ) (h1 : total = 55) (h2 : sharks = 35) (h3 : whales = 5) :
  total - sharks - whales = 15 := by
  sorry

end NUMINAMATH_CALUDE_eel_species_count_l3065_306552


namespace NUMINAMATH_CALUDE_tan_alpha_plus_beta_l3065_306570

theorem tan_alpha_plus_beta (α β : Real) 
  (h1 : Real.tan α = 1) 
  (h2 : 3 * Real.sin β = Real.sin (2 * α + β)) : 
  Real.tan (α + β) = 2 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_beta_l3065_306570


namespace NUMINAMATH_CALUDE_total_money_is_36000_l3065_306539

/-- The number of phones Vivienne has -/
def vivienne_phones : ℕ := 40

/-- The number of additional phones Aliyah has compared to Vivienne -/
def aliyah_extra_phones : ℕ := 10

/-- The price at which each phone is sold -/
def price_per_phone : ℕ := 400

/-- The total amount of money Aliyah and Vivienne have together after selling their phones -/
def total_money : ℕ := (vivienne_phones + (vivienne_phones + aliyah_extra_phones)) * price_per_phone

/-- Theorem stating that the total amount of money Aliyah and Vivienne have together is $36,000 -/
theorem total_money_is_36000 : total_money = 36000 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_36000_l3065_306539


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3065_306533

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (x - 1) * (y + 1) = 16) :
  x + y ≥ 8 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ (x₀ - 1) * (y₀ + 1) = 16 ∧ x₀ + y₀ = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3065_306533


namespace NUMINAMATH_CALUDE_tractor_circuits_l3065_306546

theorem tractor_circuits (r₁ r₂ : ℝ) (n₁ : ℕ) (h₁ : r₁ = 30) (h₂ : r₂ = 10) (h₃ : n₁ = 20) :
  ∃ n₂ : ℕ, n₂ = 60 ∧ r₁ * n₁ = r₂ * n₂ := by
  sorry

end NUMINAMATH_CALUDE_tractor_circuits_l3065_306546


namespace NUMINAMATH_CALUDE_equation_solution_l3065_306503

theorem equation_solution : ∃ x : ℝ, 3 * x - 6 = |(-20 + 5)| ∧ x = 7 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3065_306503


namespace NUMINAMATH_CALUDE_P_value_at_seven_l3065_306566

-- Define the polynomial P(x)
def P (a b c d e f : ℝ) (x : ℂ) : ℂ :=
  (3 * x^4 - 39 * x^3 + a * x^2 + b * x + c) *
  (4 * x^4 - 64 * x^3 + d * x^2 + e * x + f)

-- State the theorem
theorem P_value_at_seven 
  (a b c d e f : ℝ) 
  (h : Set.range (fun (x : ℂ) => P a b c d e f x) = {1, 2, 3, 4, 6}) : 
  P a b c d e f 7 = 69120 := by
sorry

end NUMINAMATH_CALUDE_P_value_at_seven_l3065_306566


namespace NUMINAMATH_CALUDE_line_equation_l3065_306512

/-- The ellipse E defined by the equation x^2/4 + y^2/2 = 1 -/
def E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 2 = 1}

/-- A line intersecting the ellipse E -/
def l : Set (ℝ × ℝ) := sorry

/-- Point A on the ellipse E and line l -/
def A : ℝ × ℝ := sorry

/-- Point B on the ellipse E and line l -/
def B : ℝ × ℝ := sorry

/-- The midpoint of AB is (1/2, -1) -/
axiom midpoint_AB : (A.1 + B.1) / 2 = 1/2 ∧ (A.2 + B.2) / 2 = -1

/-- Theorem stating that the equation of line l is x - 4y - 9/2 = 0 -/
theorem line_equation : l = {p : ℝ × ℝ | p.1 - 4 * p.2 - 9/2 = 0} := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l3065_306512


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l3065_306510

/-- A point (x, y) is in the third quadrant if both x and y are negative. -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- The linear function y = kx - k -/
def f (k x : ℝ) : ℝ := k * x - k

theorem linear_function_not_in_third_quadrant (k : ℝ) (h : k < 0) :
  ∀ x y : ℝ, f k x = y → ¬ in_third_quadrant x y :=
by sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l3065_306510


namespace NUMINAMATH_CALUDE_incorrect_proposition_statement_l3065_306513

theorem incorrect_proposition_statement : ∃ (p q : Prop), 
  (¬(p ∧ q)) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_proposition_statement_l3065_306513


namespace NUMINAMATH_CALUDE_weight_difference_l3065_306505

/-- Weights of different shapes in grams -/
def round_weight : ℕ := 200
def square_weight : ℕ := 300
def triangular_weight : ℕ := 150

/-- Number of weights on the left pan -/
def left_square : ℕ := 1
def left_triangular : ℕ := 2
def left_round : ℕ := 3

/-- Number of weights on the right pan -/
def right_triangular : ℕ := 1
def right_round : ℕ := 2
def right_square : ℕ := 3

/-- Total weight on the left pan -/
def left_total : ℕ := 
  left_square * square_weight + 
  left_triangular * triangular_weight + 
  left_round * round_weight

/-- Total weight on the right pan -/
def right_total : ℕ := 
  right_triangular * triangular_weight + 
  right_round * round_weight + 
  right_square * square_weight

/-- The difference in weight between the right and left pans -/
theorem weight_difference : right_total - left_total = 250 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l3065_306505


namespace NUMINAMATH_CALUDE_ant_count_in_field_l3065_306500

/-- Calculates the number of ants in a rectangular field given its dimensions in feet and ant density per square inch -/
def number_of_ants (width_feet : ℝ) (length_feet : ℝ) (ants_per_sq_inch : ℝ) : ℝ :=
  width_feet * length_feet * 144 * ants_per_sq_inch

/-- Theorem stating that a 500 by 600 feet field with 4 ants per square inch contains 172,800,000 ants -/
theorem ant_count_in_field : number_of_ants 500 600 4 = 172800000 := by
  sorry

end NUMINAMATH_CALUDE_ant_count_in_field_l3065_306500


namespace NUMINAMATH_CALUDE_volunteer_schedule_lcm_l3065_306545

theorem volunteer_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_schedule_lcm_l3065_306545


namespace NUMINAMATH_CALUDE_y_gets_0_45_per_x_rupee_l3065_306591

/-- Represents the distribution of money among three parties -/
structure MoneyDistribution where
  x : ℝ  -- amount x gets
  y : ℝ  -- amount y gets
  z : ℝ  -- amount z gets
  a : ℝ  -- amount y gets for each rupee x gets

/-- Conditions of the money distribution problem -/
def valid_distribution (d : MoneyDistribution) : Prop :=
  d.z = 0.5 * d.x ∧  -- z gets 50 paisa for each rupee x gets
  d.y = 27 ∧  -- y's share is 27 rupees
  d.x + d.y + d.z = 117 ∧  -- total amount is 117 rupees
  d.y = d.a * d.x  -- relationship between y's share and x's share

/-- Theorem stating that under the given conditions, y gets 0.45 rupees for each rupee x gets -/
theorem y_gets_0_45_per_x_rupee (d : MoneyDistribution) :
  valid_distribution d → d.a = 0.45 := by
  sorry


end NUMINAMATH_CALUDE_y_gets_0_45_per_x_rupee_l3065_306591


namespace NUMINAMATH_CALUDE_line_slope_point_sum_l3065_306544

/-- Represents a line in the form y = mx + c -/
structure Line where
  m : ℝ
  c : ℝ

/-- Checks if a point (x, y) is on the line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.c

theorem line_slope_point_sum (l : Line) :
  l.m = -5 →
  l.contains 2 3 →
  l.m + l.c = 8 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_point_sum_l3065_306544


namespace NUMINAMATH_CALUDE_det_special_matrix_is_zero_l3065_306556

open Matrix

theorem det_special_matrix_is_zero (x y z : ℝ) : 
  det ![![1, x + z, y - z],
       ![1, x + y + z, y - z],
       ![1, x + z, x + y]] = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_is_zero_l3065_306556


namespace NUMINAMATH_CALUDE_largest_number_l3065_306508

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 0.999) 
  (hb : b = 0.9099) 
  (hc : c = 0.9991) 
  (hd : d = 0.991) 
  (he : e = 0.9091) : 
  c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3065_306508


namespace NUMINAMATH_CALUDE_morgan_paid_twenty_l3065_306518

/-- Represents the cost of Morgan's lunch items and the change received --/
structure LunchTransaction where
  hamburger_cost : ℕ
  onion_rings_cost : ℕ
  smoothie_cost : ℕ
  change_received : ℕ

/-- Calculates the total amount paid by Morgan --/
def amount_paid (t : LunchTransaction) : ℕ :=
  t.hamburger_cost + t.onion_rings_cost + t.smoothie_cost + t.change_received

/-- Theorem stating that Morgan paid $20 --/
theorem morgan_paid_twenty :
  ∀ (t : LunchTransaction),
    t.hamburger_cost = 4 →
    t.onion_rings_cost = 2 →
    t.smoothie_cost = 3 →
    t.change_received = 11 →
    amount_paid t = 20 :=
by sorry

end NUMINAMATH_CALUDE_morgan_paid_twenty_l3065_306518
