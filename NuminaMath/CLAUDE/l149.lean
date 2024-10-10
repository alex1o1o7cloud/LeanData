import Mathlib

namespace cab_delay_l149_14929

theorem cab_delay (usual_time : ℝ) (speed_ratio : ℝ) (delay : ℝ) : 
  usual_time = 60 →
  speed_ratio = 5/6 →
  delay = usual_time * (1 / speed_ratio - 1) →
  delay = 12 := by
sorry

end cab_delay_l149_14929


namespace equal_area_polygons_equidecomposable_l149_14997

-- Define a polygon as a set of points in the plane
def Polygon : Type := Set (ℝ × ℝ)

-- Define the concept of area for a polygon
noncomputable def area (P : Polygon) : ℝ := sorry

-- Define equidecomposability
def equidecomposable (P Q : Polygon) : Prop := sorry

-- Theorem statement
theorem equal_area_polygons_equidecomposable (P Q : Polygon) :
  area P = area Q → equidecomposable P Q := by sorry

end equal_area_polygons_equidecomposable_l149_14997


namespace vehicle_passing_condition_min_speed_for_passing_l149_14910

-- Define the speeds and distances
def VB : ℝ := 40  -- mph
def VC : ℝ := 65  -- mph
def dist_AB : ℝ := 100  -- ft
def dist_BC : ℝ := 250  -- ft

-- Define the theorem
theorem vehicle_passing_condition (VA : ℝ) :
  VA > 2 →
  (dist_AB / (VA + VB)) < (dist_BC / (VB + VC)) :=
by
  sorry

-- Define the main theorem that answers the original question
theorem min_speed_for_passing :
  ∃ (VA : ℝ), VA > 2 ∧
  ∀ (VA' : ℝ), VA' > VA →
  (dist_AB / (VA' + VB)) < (dist_BC / (VB + VC)) :=
by
  sorry

end vehicle_passing_condition_min_speed_for_passing_l149_14910


namespace cork_price_calculation_l149_14922

/-- The price of a bottle of wine with a cork -/
def bottle_with_cork : ℚ := 2.10

/-- The additional cost of a bottle without a cork compared to the cork price -/
def additional_cost : ℚ := 2.00

/-- The discount rate for the cork when buying in large quantities -/
def cork_discount : ℚ := 0.12

/-- The price of the cork before discount -/
def cork_price : ℚ := bottle_with_cork - (bottle_with_cork - additional_cost) / 2

/-- The discounted price of the cork -/
def discounted_cork_price : ℚ := cork_price * (1 - cork_discount)

theorem cork_price_calculation :
  discounted_cork_price = 0.044 := by sorry

end cork_price_calculation_l149_14922


namespace ivy_coverage_l149_14931

/-- The amount of ivy Cary strips each day, in feet -/
def daily_strip : ℕ := 6

/-- The amount of ivy that grows back each night, in feet -/
def nightly_growth : ℕ := 2

/-- The number of days it takes Cary to strip all the ivy -/
def days_to_strip : ℕ := 10

/-- The net amount of ivy stripped per day, in feet -/
def net_strip_per_day : ℕ := daily_strip - nightly_growth

/-- The total amount of ivy covering the tree, in feet -/
def total_ivy : ℕ := net_strip_per_day * days_to_strip

theorem ivy_coverage : total_ivy = 40 := by
  sorry

end ivy_coverage_l149_14931


namespace grade_change_impossibility_l149_14986

theorem grade_change_impossibility : ¬ ∃ (n₁ n₂ n₃ n₄ : ℤ),
  2 * n₁ + n₂ - 2 * n₃ - n₄ = 27 ∧
  -n₁ + 2 * n₂ + n₃ - 2 * n₄ = -27 :=
sorry

end grade_change_impossibility_l149_14986


namespace dot_product_theorem_l149_14998

def a : ℝ × ℝ := (1, 3)

theorem dot_product_theorem (b : ℝ × ℝ) 
  (h1 : Real.sqrt ((b.1 - 1)^2 + (b.2 - 3)^2) = Real.sqrt 10)
  (h2 : Real.sqrt (b.1^2 + b.2^2) = 2) :
  (2 * a.1 + b.1) * (a.1 - b.1) + (2 * a.2 + b.2) * (a.2 - b.2) = 14 := by
  sorry

end dot_product_theorem_l149_14998


namespace rectangleWithHoleAreaTheorem_l149_14925

/-- The area of a rectangle with a hole, given the dimensions of both rectangles -/
def rectangleWithHoleArea (x : ℝ) : ℝ :=
  let largeRectLength := x + 7
  let largeRectWidth := x + 5
  let holeLength := 2*x - 3
  let holeWidth := x - 2
  (largeRectLength * largeRectWidth) - (holeLength * holeWidth)

/-- Theorem stating that the area of the rectangle with a hole is equal to -x^2 + 19x + 29 -/
theorem rectangleWithHoleAreaTheorem (x : ℝ) :
  rectangleWithHoleArea x = -x^2 + 19*x + 29 := by
  sorry

end rectangleWithHoleAreaTheorem_l149_14925


namespace quadratic_roots_relation_l149_14926

theorem quadratic_roots_relation (A B C : ℝ) (r s p q : ℝ) : 
  (A * r^2 + B * r + C = 0) →
  (A * s^2 + B * s + C = 0) →
  (r^2)^2 + p * r^2 + q = 0 →
  (s^2)^2 + p * s^2 + q = 0 →
  p = (2 * A * C - B^2) / A^2 := by
sorry

end quadratic_roots_relation_l149_14926


namespace min_value_expression_l149_14982

theorem min_value_expression (x y : ℝ) : (3*x*y - 1)^2 + (x - y)^2 ≥ 1 := by
  sorry

end min_value_expression_l149_14982


namespace base_three_digit_difference_l149_14963

/-- The number of digits in the base-b representation of a positive integer n -/
def numDigits (n : ℕ+) (b : ℕ) : ℕ :=
  Nat.log b n + 1

/-- Theorem: The number of digits in the base-3 representation of 1500
    is exactly 1 more than the number of digits in the base-3 representation of 300 -/
theorem base_three_digit_difference :
  numDigits 1500 3 = numDigits 300 3 + 1 := by
  sorry

end base_three_digit_difference_l149_14963


namespace simplify_fraction_product_l149_14968

theorem simplify_fraction_product : 
  4 * (18 / 5) * (25 / -45) * (10 / 8) = -10 := by
  sorry

end simplify_fraction_product_l149_14968


namespace water_mixture_percentage_l149_14967

/-- Proves that adding a specific amount of water to a given mixture results in a new mixture with the expected water percentage. -/
theorem water_mixture_percentage 
  (initial_volume : ℝ) 
  (initial_water_percentage : ℝ) 
  (added_water : ℝ) 
  (h1 : initial_volume = 200)
  (h2 : initial_water_percentage = 20)
  (h3 : added_water = 13.333333333333334)
  : (initial_water_percentage / 100 * initial_volume + added_water) / (initial_volume + added_water) * 100 = 25 := by
  sorry

end water_mixture_percentage_l149_14967


namespace orange_price_problem_l149_14990

/-- Proof of the orange price problem --/
theorem orange_price_problem 
  (apple_price : ℚ) 
  (total_fruits : ℕ) 
  (initial_avg_price : ℚ) 
  (oranges_removed : ℕ) 
  (final_avg_price : ℚ) 
  (h1 : apple_price = 40/100)
  (h2 : total_fruits = 10)
  (h3 : initial_avg_price = 56/100)
  (h4 : oranges_removed = 6)
  (h5 : final_avg_price = 50/100) :
  ∃ (orange_price : ℚ), orange_price = 60/100 := by
sorry


end orange_price_problem_l149_14990


namespace probability_one_defective_six_two_l149_14942

/-- The probability of selecting exactly one defective product from a set of products -/
def probability_one_defective (total : ℕ) (defective : ℕ) : ℚ :=
  let qualified := total - defective
  (defective.choose 1 * qualified.choose 1 : ℚ) / total.choose 2

/-- Given 6 products with 2 defective ones, the probability of selecting exactly one defective product is 8/15 -/
theorem probability_one_defective_six_two :
  probability_one_defective 6 2 = 8 / 15 := by
  sorry

#eval probability_one_defective 6 2

end probability_one_defective_six_two_l149_14942


namespace bridesmaids_dresses_completion_time_l149_14996

def dress_hours : List Nat := [15, 18, 20, 22, 24, 26, 28]
def hours_per_week : Nat := 5

theorem bridesmaids_dresses_completion_time : 
  ∃ (total_hours : Nat),
    total_hours = dress_hours.sum ∧
    (total_hours / hours_per_week : ℚ) ≤ 31 ∧
    31 < (total_hours / hours_per_week : ℚ) + 1 :=
by sorry

end bridesmaids_dresses_completion_time_l149_14996


namespace production_increase_l149_14946

theorem production_increase (n : ℕ) (old_avg new_avg : ℚ) (today_production : ℚ) : 
  n = 19 →
  old_avg = 50 →
  new_avg = 52 →
  today_production = n * old_avg + today_production →
  (n + 1) * new_avg = n * old_avg + today_production →
  today_production = 90 := by
  sorry

end production_increase_l149_14946


namespace ellipse_perpendicular_chord_bounds_l149_14978

/-- Given an ellipse (x²/a²) + (y²/b²) = 1 with a > b > 0, for any two points A and B on the ellipse
    such that OA ⊥ OB, the distance |AB| satisfies (ab / √(a² + b²)) ≤ |AB| ≤ √(a² + b²) -/
theorem ellipse_perpendicular_chord_bounds (a b : ℝ) (ha : 0 < b) (hab : b < a) :
  ∀ (A B : ℝ × ℝ),
    (A.1^2 / a^2 + A.2^2 / b^2 = 1) →
    (B.1^2 / a^2 + B.2^2 / b^2 = 1) →
    (A.1 * B.1 + A.2 * B.2 = 0) →
    (a * b / Real.sqrt (a^2 + b^2) ≤ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) ∧
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt (a^2 + b^2)) := by
  sorry

end ellipse_perpendicular_chord_bounds_l149_14978


namespace machine_input_l149_14958

/-- A machine that processes numbers -/
def Machine (x : ℤ) : ℤ := x + 15 - 6

/-- Theorem: If the machine outputs 35, the input must have been 26 -/
theorem machine_input (x : ℤ) : Machine x = 35 → x = 26 := by
  sorry

end machine_input_l149_14958


namespace negation_of_implication_l149_14909

theorem negation_of_implication (p q : Prop) : 
  ¬(p → q) ↔ p ∧ ¬q := by sorry

end negation_of_implication_l149_14909


namespace existence_of_three_quadratics_l149_14961

theorem existence_of_three_quadratics : ∃ (f₁ f₂ f₃ : ℝ → ℝ),
  (∃ x₁, f₁ x₁ = 0) ∧
  (∃ x₂, f₂ x₂ = 0) ∧
  (∃ x₃, f₃ x₃ = 0) ∧
  (∀ x, (f₁ x + f₂ x) ≠ 0) ∧
  (∀ x, (f₁ x + f₃ x) ≠ 0) ∧
  (∀ x, (f₂ x + f₃ x) ≠ 0) ∧
  (∀ x, f₁ x = (x - 1)^2) ∧
  (∀ x, f₂ x = x^2) ∧
  (∀ x, f₃ x = (x - 2)^2) :=
by sorry

end existence_of_three_quadratics_l149_14961


namespace num_tuba_players_l149_14970

/-- The weight carried by each trumpet or clarinet player -/
def trumpet_clarinet_weight : ℕ := 5

/-- The weight carried by each trombone player -/
def trombone_weight : ℕ := 10

/-- The weight carried by each tuba player -/
def tuba_weight : ℕ := 20

/-- The weight carried by each drum player -/
def drum_weight : ℕ := 15

/-- The number of trumpet players -/
def num_trumpets : ℕ := 6

/-- The number of clarinet players -/
def num_clarinets : ℕ := 9

/-- The number of trombone players -/
def num_trombones : ℕ := 8

/-- The number of drum players -/
def num_drums : ℕ := 2

/-- The total weight carried by the marching band -/
def total_weight : ℕ := 245

/-- Theorem: The number of tuba players in the marching band is 3 -/
theorem num_tuba_players : 
  ∃ (n : ℕ), n * tuba_weight = 
    total_weight - 
    (num_trumpets * trumpet_clarinet_weight + 
     num_clarinets * trumpet_clarinet_weight + 
     num_trombones * trombone_weight + 
     num_drums * drum_weight) ∧ 
  n = 3 := by
  sorry

end num_tuba_players_l149_14970


namespace square_area_16m_l149_14944

/-- The area of a square with side length 16 meters is 256 square meters. -/
theorem square_area_16m (side_length : ℝ) (h : side_length = 16) : 
  side_length * side_length = 256 := by
  sorry

end square_area_16m_l149_14944


namespace polynomial_equality_l149_14993

theorem polynomial_equality (d e c : ℝ) : 
  (∀ x : ℝ, (6 * x^2 - 5 * x + 10/3) * (d * x^2 + e * x + c) = 
    18 * x^4 - 5 * x^3 + 15 * x^2 - (50/3) * x + 45/3) → 
  c = 4.5 := by
  sorry

end polynomial_equality_l149_14993


namespace power_mod_thirteen_l149_14954

theorem power_mod_thirteen : 777^777 % 13 = 1 := by
  sorry

end power_mod_thirteen_l149_14954


namespace computation_proof_l149_14983

theorem computation_proof : 55 * 1212 - 15 * 1212 = 48480 := by
  sorry

end computation_proof_l149_14983


namespace rachel_homework_difference_l149_14957

/-- Rachel's homework problem -/
theorem rachel_homework_difference :
  ∀ (math_pages reading_pages biology_pages : ℕ),
    math_pages = 9 →
    reading_pages = 2 →
    biology_pages = 96 →
    math_pages - reading_pages = 7 :=
by
  sorry

end rachel_homework_difference_l149_14957


namespace thirtieth_triangular_and_difference_l149_14960

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem thirtieth_triangular_and_difference :
  (triangular_number 30 = 465) ∧
  (triangular_number 30 - triangular_number 29 = 30) := by
  sorry

end thirtieth_triangular_and_difference_l149_14960


namespace parsnip_box_ratio_l149_14913

/-- Represents the number of parsnips in a full box -/
def full_box_capacity : ℕ := 20

/-- Represents the total number of boxes in an average harvest -/
def total_boxes : ℕ := 20

/-- Represents the total number of parsnips in an average harvest -/
def total_parsnips : ℕ := 350

/-- Represents the number of full boxes -/
def full_boxes : ℕ := 15

/-- Represents the number of half-full boxes -/
def half_full_boxes : ℕ := total_boxes - full_boxes

theorem parsnip_box_ratio :
  (full_boxes : ℚ) / total_boxes = 3 / 4 ∧
  full_boxes + half_full_boxes = total_boxes ∧
  full_boxes * full_box_capacity + half_full_boxes * (full_box_capacity / 2) = total_parsnips :=
by sorry

end parsnip_box_ratio_l149_14913


namespace sum_of_coefficients_equals_negative_29_l149_14934

theorem sum_of_coefficients_equals_negative_29 :
  let p (x : ℝ) := 5 * (2 * x^8 - 9 * x^3 + 6) - 4 * (x^6 + 8 * x^3 - 3)
  (p 1) = -29 :=
by
  sorry

end sum_of_coefficients_equals_negative_29_l149_14934


namespace quadratic_root_problem_l149_14930

/-- If the roots of the quadratic equation 5x^2 + 4x + k are (-4 ± i√379) / 10, then k = 19.75 -/
theorem quadratic_root_problem (k : ℝ) : 
  (∀ x : ℂ, 5 * x^2 + 4 * x + k = 0 ↔ x = (-4 + ℍ * Real.sqrt 379) / 10 ∨ x = (-4 - ℍ * Real.sqrt 379) / 10) →
  k = 19.75 :=
by sorry

end quadratic_root_problem_l149_14930


namespace new_drive_size_l149_14936

/-- Calculates the size of a new external drive based on initial drive conditions and file operations -/
theorem new_drive_size
  (initial_free : ℝ)
  (initial_used : ℝ)
  (deleted_size : ℝ)
  (new_files_size : ℝ)
  (new_free_space : ℝ)
  (h1 : initial_free = 2.4)
  (h2 : initial_used = 12.6)
  (h3 : deleted_size = 4.6)
  (h4 : new_files_size = 2)
  (h5 : new_free_space = 10) :
  initial_used - deleted_size + new_files_size + new_free_space = 20 := by
  sorry

#check new_drive_size

end new_drive_size_l149_14936


namespace sum_of_fractions_l149_14966

theorem sum_of_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 8 = (5 : ℚ) / 8 := by
  sorry

end sum_of_fractions_l149_14966


namespace rectangular_plot_minus_circular_garden_l149_14918

/-- The area of a rectangular plot minus a circular garden --/
theorem rectangular_plot_minus_circular_garden :
  let rectangle_length : ℝ := 8
  let rectangle_width : ℝ := 12
  let circle_radius : ℝ := 3
  let rectangle_area := rectangle_length * rectangle_width
  let circle_area := π * circle_radius ^ 2
  rectangle_area - circle_area = 96 - 9 * π := by sorry

end rectangular_plot_minus_circular_garden_l149_14918


namespace quadratic_inequality_solution_set_l149_14919

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end quadratic_inequality_solution_set_l149_14919


namespace problem_statement_l149_14938

def p (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, 2*x - 2 ≥ m^2 - 3*m

def q (m : ℝ) : Prop := ∃ x ∈ Set.Icc (-1) 1, m ≤ x

theorem problem_statement (m : ℝ) :
  (p m ↔ m ∈ Set.Icc 1 2) ∧
  (¬(p m ∧ q m) ∧ (p m ∨ q m) ↔ m < 1 ∨ (1 < m ∧ m ≤ 2)) :=
by sorry

end problem_statement_l149_14938


namespace flower_bed_properties_l149_14988

/-- Represents a rectangular flower bed with specific properties -/
structure FlowerBed where
  length : ℝ
  width : ℝ
  area : ℝ
  theta : ℝ

/-- Theorem about the properties of a specific flower bed -/
theorem flower_bed_properties :
  ∃ (fb : FlowerBed),
    fb.area = 50 ∧
    fb.width = (2/3) * fb.length ∧
    Real.tan fb.theta = (fb.length - fb.width) / (fb.length + fb.width) ∧
    fb.length = 75 ∧
    fb.width = 50 ∧
    Real.tan fb.theta = 1/5 := by
  sorry


end flower_bed_properties_l149_14988


namespace shark_sightings_total_l149_14995

/-- The number of shark sightings in Daytona Beach -/
def daytona_beach_sightings : ℕ := sorry

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := 24

/-- Cape May has 8 less than double the number of shark sightings of Daytona Beach -/
axiom cape_may_relation : cape_may_sightings = 2 * daytona_beach_sightings - 8

/-- The total number of shark sightings in both locations -/
def total_sightings : ℕ := cape_may_sightings + daytona_beach_sightings

theorem shark_sightings_total : total_sightings = 40 := by
  sorry

end shark_sightings_total_l149_14995


namespace remainder_problem_l149_14981

theorem remainder_problem (x : ℕ+) : (6 * x.val) % 9 = 3 → x.val % 9 = 5 := by
  sorry

end remainder_problem_l149_14981


namespace sector_angle_measure_l149_14992

/-- Given a circular sector with arc length and area both equal to 5,
    prove that the radian measure of its central angle is 5/2 -/
theorem sector_angle_measure (r : ℝ) (α : ℝ) 
    (h1 : α * r = 5)  -- arc length formula
    (h2 : (1/2) * α * r^2 = 5)  -- sector area formula
    : α = 5/2 := by
  sorry

end sector_angle_measure_l149_14992


namespace triangle_isosceles_l149_14927

/-- A triangle with sides a, b, and c satisfying the given equation is isosceles with c as the base -/
theorem triangle_isosceles (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : 1/a - 1/b + 1/c = 1/(a-b+c)) : a = c :=
sorry

end triangle_isosceles_l149_14927


namespace problem_l149_14950

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 2 * Real.cos x + 1

theorem problem (a b c : ℝ) (h : ∀ x : ℝ, a * f x + b * f (x - c) = 1) :
  b * Real.cos c / a = -1 := by sorry

end problem_l149_14950


namespace roots_cubic_reciprocal_sum_l149_14901

/-- Given a quadratic equation px^2 + qx + m = 0 with roots r and s,
    prove that 1/r^3 + 1/s^3 = (-q^3 + 3qm) / m^3 -/
theorem roots_cubic_reciprocal_sum (p q m : ℝ) (hp : p ≠ 0) (hm : m ≠ 0) :
  ∃ (r s : ℝ), (p * r^2 + q * r + m = 0) ∧ 
               (p * s^2 + q * s + m = 0) ∧ 
               (1 / r^3 + 1 / s^3 = (-q^3 + 3*q*m) / m^3) := by
  sorry

end roots_cubic_reciprocal_sum_l149_14901


namespace rectangle_diagonal_l149_14900

/-- Given a rectangle with perimeter 80 meters and length-to-width ratio of 5:2,
    prove that its diagonal length is sqrt(46400)/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 80) →
  (length / width = 5 / 2) →
  Real.sqrt (length^2 + width^2) = Real.sqrt 46400 / 7 := by
  sorry

end rectangle_diagonal_l149_14900


namespace tangent_circles_radii_l149_14945

/-- Two externally tangent circles with specific properties -/
structure TangentCircles where
  r₁ : ℝ  -- radius of the smaller circle
  r₂ : ℝ  -- radius of the larger circle
  h₁ : r₂ = r₁ + 5  -- difference between radii is 5
  h₂ : ∃ (d : ℝ), d = 2.4 * r₁ ∧ d^2 + r₁^2 = (r₂ - r₁)^2  -- distance property

/-- The radii of the two circles are 4 and 9 -/
theorem tangent_circles_radii (c : TangentCircles) : c.r₁ = 4 ∧ c.r₂ = 9 :=
  sorry

end tangent_circles_radii_l149_14945


namespace remaining_quarters_l149_14975

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.50
def jeans_cost : ℚ := 11.50
def quarter_value : ℚ := 0.25

theorem remaining_quarters : 
  (initial_amount - (pizza_cost + soda_cost + jeans_cost)) / quarter_value = 97 := by
  sorry

end remaining_quarters_l149_14975


namespace sum_of_primes_even_l149_14949

/-- A number is prime if it's greater than 1 and has no positive divisors other than 1 and itself -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem sum_of_primes_even 
  (A B C : ℕ) 
  (hA : isPrime A) 
  (hB : isPrime B) 
  (hC : isPrime C) 
  (hAB_minus : isPrime (A - B)) 
  (hAB_plus : isPrime (A + B)) 
  (hABC : isPrime (A + B + C)) : 
  Even (A + B + C + (A - B) + (A + B) + (A + B + C)) := by
sorry

end sum_of_primes_even_l149_14949


namespace current_speed_l149_14989

/-- Given a boat that moves upstream at 1 km in 40 minutes and downstream at 1 km in 12 minutes,
    prove that the speed of the current is 1.75 km/h. -/
theorem current_speed (upstream_speed : ℝ) (downstream_speed : ℝ)
    (h1 : upstream_speed = 1 / (40 / 60))  -- 1 km in 40 minutes converted to km/h
    (h2 : downstream_speed = 1 / (12 / 60))  -- 1 km in 12 minutes converted to km/h
    : (downstream_speed - upstream_speed) / 2 = 1.75 := by
  sorry

end current_speed_l149_14989


namespace probability_of_one_red_ball_l149_14943

/-- The probability of drawing exactly one red ball from a bag containing 2 yellow balls, 3 red balls, and 5 white balls is 3/10. -/
theorem probability_of_one_red_ball (yellow_balls red_balls white_balls : ℕ) 
  (h_yellow : yellow_balls = 2)
  (h_red : red_balls = 3)
  (h_white : white_balls = 5) : 
  (red_balls : ℚ) / (yellow_balls + red_balls + white_balls) = 3 / 10 := by
  sorry

end probability_of_one_red_ball_l149_14943


namespace simplify_nested_roots_l149_14947

theorem simplify_nested_roots (a : ℝ) (ha : a > 0) :
  (((a^16)^(1/8))^(1/4))^12 * (((a^16)^(1/4))^(1/8))^12 = a^12 := by
  sorry

end simplify_nested_roots_l149_14947


namespace boys_ratio_in_class_l149_14953

theorem boys_ratio_in_class (n_boys n_girls : ℕ) (h_prob : n_boys / (n_boys + n_girls) = 2/3 * (n_girls / (n_boys + n_girls))) :
  n_boys / (n_boys + n_girls) = 2/5 := by
  sorry

end boys_ratio_in_class_l149_14953


namespace logarithm_expression_equality_l149_14976

theorem logarithm_expression_equality : 
  9^(Real.log 2 / Real.log 3) - 4 * (Real.log 3 / Real.log 4) * (Real.log 8 / Real.log 27) + 
  (1/3) * (Real.log 8 / Real.log 6) - 2 * (Real.log (Real.sqrt 3) / Real.log (1/6)) = 3 := by
  sorry

end logarithm_expression_equality_l149_14976


namespace integer_divisibility_problem_l149_14962

theorem integer_divisibility_problem (n : ℤ) :
  (5 ∣ (3 * n - 2)) ∧ (7 ∣ (2 * n + 1)) ↔ ∃ m : ℤ, n = 35 * m + 24 := by
  sorry

end integer_divisibility_problem_l149_14962


namespace age_difference_proof_l149_14977

def elder_age : ℕ := 30

theorem age_difference_proof (younger_age : ℕ) 
  (h : elder_age - 6 = 3 * (younger_age - 6)) : 
  elder_age - younger_age = 16 := by
  sorry

end age_difference_proof_l149_14977


namespace parallel_vectors_x_value_l149_14959

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let m : ℝ × ℝ := (1, 2)
  let n : ℝ → ℝ × ℝ := λ x ↦ (x, 2 - 2*x)
  ∀ x : ℝ, are_parallel m (n x) → x = 1/2 := by
  sorry

end parallel_vectors_x_value_l149_14959


namespace elevator_weight_problem_l149_14951

/-- Given 6 people in an elevator, if a 7th person weighing 133 lbs enters
    and the new average weight becomes 151 lbs, then the initial average
    weight was 154 lbs. -/
theorem elevator_weight_problem :
  ∀ (initial_average : ℝ),
  (6 * initial_average + 133) / 7 = 151 →
  initial_average = 154 :=
by
  sorry

end elevator_weight_problem_l149_14951


namespace equilateral_triangle_product_l149_14923

/-- Given that (0,0), (a,19), and (b,61) are vertices of an equilateral triangle, prove that ab = 7760/9 -/
theorem equilateral_triangle_product (a b : ℝ) : 
  (∀ (z : ℂ), z^3 = 1 ∧ z ≠ 1 → (a + 19*I) * z = b + 61*I) → 
  a * b = 7760 / 9 := by
sorry


end equilateral_triangle_product_l149_14923


namespace arcsin_neg_half_equals_neg_pi_sixth_l149_14917

theorem arcsin_neg_half_equals_neg_pi_sixth : 
  Real.arcsin (-0.5) = -π/6 := by sorry

end arcsin_neg_half_equals_neg_pi_sixth_l149_14917


namespace max_type_c_test_tubes_l149_14933

/-- Represents the number of test tubes of each type -/
structure TestTubes where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the solution percentages are valid -/
def validSolution (t : TestTubes) : Prop :=
  10 * t.a + 20 * t.b + 90 * t.c = 2017 * (t.a + t.b + t.c)

/-- Checks if the total number of test tubes is 1000 -/
def totalIs1000 (t : TestTubes) : Prop :=
  t.a + t.b + t.c = 1000

/-- Checks if test tubes of the same type are not used consecutively -/
def noConsecutiveSameType (t : TestTubes) : Prop :=
  7 * t.c ≤ 517 ∧ 8 * t.c ≥ 518 ∧ t.c ≤ 500

/-- Theorem: The maximum number of type C test tubes is 73 -/
theorem max_type_c_test_tubes :
  ∃ (t : TestTubes),
    validSolution t ∧
    totalIs1000 t ∧
    noConsecutiveSameType t ∧
    (∀ (t' : TestTubes),
      validSolution t' ∧ totalIs1000 t' ∧ noConsecutiveSameType t' →
      t'.c ≤ t.c) ∧
    t.c = 73 :=
  sorry

end max_type_c_test_tubes_l149_14933


namespace quarter_circle_roll_path_length_l149_14921

/-- The length of the path traveled by point F when rolling a quarter-circle -/
theorem quarter_circle_roll_path_length 
  (radius : ℝ) 
  (h_radius : radius = 3 / Real.pi) : 
  let path_length := 3 * (Real.pi * radius / 2)
  path_length = 4.5 := by sorry

end quarter_circle_roll_path_length_l149_14921


namespace min_value_a_l149_14969

theorem min_value_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 0) :
  (∀ x y, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) →
  a ≥ Real.sqrt 2 :=
by sorry

end min_value_a_l149_14969


namespace root_sum_reciprocal_difference_l149_14973

-- Define the polynomial
def f (x : ℝ) : ℝ := 20 * x^3 - 40 * x^2 + 18 * x - 1

-- Define the roots
variable (a b c : ℝ)

-- State the theorem
theorem root_sum_reciprocal_difference (ha : f a = 0) (hb : f b = 0) (hc : f c = 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (ha_bounds : 0 < a ∧ a < 1) (hb_bounds : 0 < b ∧ b < 1) (hc_bounds : 0 < c ∧ c < 1) :
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1 := by
  sorry

end root_sum_reciprocal_difference_l149_14973


namespace product_of_polynomials_l149_14935

theorem product_of_polynomials (p x y : ℝ) : 
  (2 * p^2 - 5 * p + x) * (5 * p^2 + y * p - 10) = 10 * p^4 + 5 * p^3 - 65 * p^2 + 40 * p + 40 →
  x + y = -6.5 := by
sorry

end product_of_polynomials_l149_14935


namespace kendall_driving_distance_l149_14948

theorem kendall_driving_distance (distance_with_mother distance_with_father : Real) 
  (h1 : distance_with_mother = 0.17)
  (h2 : distance_with_father = 0.5) : 
  distance_with_mother + distance_with_father = 0.67 := by
  sorry

end kendall_driving_distance_l149_14948


namespace prob_A_plus_B_complement_l149_14991

-- Define the sample space
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define event A
def A : Finset Nat := {2, 4}

-- Define event B
def B : Finset Nat := {1, 2, 3, 4, 5}

-- Define the complement of B
def B_complement : Finset Nat := Ω \ B

-- Define the probability measure
def P (E : Finset Nat) : Rat := (E.card : Rat) / (Ω.card : Rat)

-- State the theorem
theorem prob_A_plus_B_complement : P (A ∪ B_complement) = 1/2 := by
  sorry

end prob_A_plus_B_complement_l149_14991


namespace isosceles_triangles_angle_l149_14902

/-- Isosceles triangle -/
structure IsoscelesTriangle (P Q R : ℝ × ℝ) :=
  (isosceles : dist P Q = dist Q R)

/-- Similar triangles -/
def SimilarTriangles (P Q R P' Q' R' : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    dist P Q = k * dist P' Q' ∧
    dist Q R = k * dist Q' R' ∧
    dist R P = k * dist R' P'

/-- Point lies on line segment -/
def OnSegment (P A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

/-- Point lies on line extension -/
def OnExtension (P A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ P = (1 - t) • A + t • B

/-- Perpendicular line segments -/
def Perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (R.1 - P.1) * (S.1 - Q.1) + (R.2 - P.2) * (S.2 - Q.2) = 0

/-- Angle measure -/
def AngleMeasure (P Q R : ℝ × ℝ) : ℝ :=
  sorry

theorem isosceles_triangles_angle (A B C A₁ B₁ C₁ : ℝ × ℝ) :
  IsoscelesTriangle A B C →
  IsoscelesTriangle A₁ B₁ C₁ →
  SimilarTriangles A B C A₁ B₁ C₁ →
  dist A C / dist A₁ C₁ = 5 / Real.sqrt 3 →
  OnSegment A₁ A C →
  OnSegment B₁ B C →
  OnExtension C₁ A B →
  Perpendicular A₁ B₁ B C →
  AngleMeasure A B C = 120 * π / 180 :=
sorry

end isosceles_triangles_angle_l149_14902


namespace derivative_cos_at_pi_12_l149_14985

/-- Given a function f(x) = cos(2x + π/3), prove that its derivative at x = π/12 is -2. -/
theorem derivative_cos_at_pi_12 (f : ℝ → ℝ) (h : ∀ x, f x = Real.cos (2 * x + π / 3)) :
  deriv f (π / 12) = -2 := by
  sorry

end derivative_cos_at_pi_12_l149_14985


namespace haley_cousins_count_l149_14994

/-- The number of origami papers Haley has to give away -/
def total_papers : ℕ := 48

/-- The number of origami papers each cousin receives -/
def papers_per_cousin : ℕ := 8

/-- Haley's number of cousins -/
def num_cousins : ℕ := total_papers / papers_per_cousin

theorem haley_cousins_count : num_cousins = 6 := by
  sorry

end haley_cousins_count_l149_14994


namespace fires_put_out_l149_14912

/-- The number of fires Doug put out -/
def doug_fires : ℕ := 20

/-- The number of fires Kai put out -/
def kai_fires : ℕ := 3 * doug_fires

/-- The number of fires Eli put out -/
def eli_fires : ℕ := kai_fires / 2

/-- The total number of fires put out by Doug, Kai, and Eli -/
def total_fires : ℕ := doug_fires + kai_fires + eli_fires

theorem fires_put_out : total_fires = 110 := by
  sorry

end fires_put_out_l149_14912


namespace probability_odd_product_sum_div_5_l149_14939

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 20 ∧ 1 ≤ b ∧ b ≤ 20 ∧ a ≠ b ∧
  is_odd a ∧ is_odd b ∧ is_divisible_by_5 (a + b)

def total_pairs : ℕ := 190

def valid_pairs : ℕ := 6

theorem probability_odd_product_sum_div_5 :
  (valid_pairs : ℚ) / total_pairs = 3 / 95 := by sorry

end probability_odd_product_sum_div_5_l149_14939


namespace propositions_truthfulness_l149_14972

-- Define the properties
def isPositiveEven (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

-- Theorem statement
theorem propositions_truthfulness :
  (∃ n : ℕ, isPositiveEven n ∧ isPrime n) ∧
  (∃ n : ℕ, ¬isPrime n ∧ ¬isPositiveEven n) ∧
  (∃ n : ℕ, ¬isPositiveEven n ∧ ¬isPrime n) ∧
  (∀ n : ℕ, isPrime n → ¬isPositiveEven n) :=
sorry

end propositions_truthfulness_l149_14972


namespace equal_volume_cans_l149_14903

/-- Represents a cylindrical can with radius and height -/
structure Can where
  radius : ℝ
  height : ℝ

/-- Theorem stating the relation between two cans with equal volume -/
theorem equal_volume_cans (can1 can2 : Can) 
  (h_volume : can1.radius ^ 2 * can1.height = can2.radius ^ 2 * can2.height)
  (h_height : can2.height = 4 * can1.height)
  (h_narrow_radius : can1.radius = 10) :
  can2.radius = 20 := by
  sorry

end equal_volume_cans_l149_14903


namespace average_height_is_12_l149_14956

def plant_heights (h1 h2 h3 h4 : ℝ) : Prop :=
  h1 = 27 ∧ h3 = 9 ∧
  ((h2 = h1 / 3 ∨ h2 = h1 * 3) ∧
   (h3 = h2 / 3 ∨ h3 = h2 * 3) ∧
   (h4 = h3 / 3 ∨ h4 = h3 * 3))

theorem average_height_is_12 (h1 h2 h3 h4 : ℝ) :
  plant_heights h1 h2 h3 h4 → (h1 + h2 + h3 + h4) / 4 = 12 :=
by
  sorry

end average_height_is_12_l149_14956


namespace five_students_two_groups_l149_14906

/-- The number of ways to assign students to groups -/
def assignment_count (num_students : ℕ) (num_groups : ℕ) : ℕ :=
  num_groups ^ num_students

/-- Theorem: There are 32 ways to assign 5 students to 2 groups -/
theorem five_students_two_groups :
  assignment_count 5 2 = 32 := by
  sorry

end five_students_two_groups_l149_14906


namespace max_cookies_andy_l149_14920

/-- Represents the number of cookies eaten by each sibling -/
structure CookieDistribution where
  andy : ℕ
  alexa : ℕ
  john : ℕ

/-- Checks if a cookie distribution is valid according to the problem conditions -/
def isValidDistribution (d : CookieDistribution) : Prop :=
  d.alexa = 2 * d.andy + 2 ∧
  d.john = d.andy - 3 ∧
  d.andy + d.alexa + d.john = 30

/-- Theorem stating that the maximum number of cookies Andy can eat is 7 -/
theorem max_cookies_andy :
  ∀ d : CookieDistribution, isValidDistribution d → d.andy ≤ 7 :=
by sorry

end max_cookies_andy_l149_14920


namespace queen_center_probability_queen_center_probability_2004_l149_14916

/-- Probability of queen being in center after n moves -/
def prob_queen_center (n : ℕ) : ℚ :=
  1/3 + 2/3 * (-1/2)^n

/-- Initial configuration with queen in center -/
def initial_config : List Char := ['R', 'Q', 'R']

/-- Theorem stating the probability of queen being in center after n moves -/
theorem queen_center_probability (n : ℕ) : 
  prob_queen_center n = 1/3 + 2/3 * (-1/2)^n :=
sorry

/-- Corollary for the specific case of 2004 moves -/
theorem queen_center_probability_2004 : 
  prob_queen_center 2004 = 1/3 + 1/(3 * 2^2003) :=
sorry

end queen_center_probability_queen_center_probability_2004_l149_14916


namespace inequality_proof_l149_14937

theorem inequality_proof (x : ℝ) (n : ℕ) (hx : x > 1) (hn : n > 1) :
  1 + (x - 1) / (n * x) < x^(1/n) ∧ x^(1/n) < 1 + (x - 1) / n :=
by sorry

end inequality_proof_l149_14937


namespace range_of_2a_minus_b_l149_14955

theorem range_of_2a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 5) (hb : 5 < b ∧ b < 12) :
  -10 < 2 * a - b ∧ 2 * a - b < 5 := by
  sorry

end range_of_2a_minus_b_l149_14955


namespace problem_solution_l149_14924

theorem problem_solution : ∀ x y : ℝ,
  x = 88 * (1 + 0.3) →
  y = x * (1 - 0.15) →
  y = 97.24 := by
sorry

end problem_solution_l149_14924


namespace no_complex_numbers_satisfying_condition_l149_14932

theorem no_complex_numbers_satisfying_condition : ¬∃ (a b c : ℂ) (h : ℕ), 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ 
  (∀ (k l m : ℤ), (abs k + abs l + abs m ≥ 1996) → 
    Complex.abs (k • a + l • b + m • c) > 1 / h) :=
by sorry

end no_complex_numbers_satisfying_condition_l149_14932


namespace intersection_implies_outside_circle_l149_14941

theorem intersection_implies_outside_circle (a b : ℝ) :
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) →
  a^2 + b^2 > 1 :=
by sorry

end intersection_implies_outside_circle_l149_14941


namespace bus_rows_theorem_l149_14971

/-- Represents the state of passengers on a bus -/
structure BusState where
  initial : Nat
  first_stop_board : Nat
  first_stop_leave : Nat
  second_stop_board : Nat
  second_stop_leave : Nat
  empty_seats : Nat
  seats_per_row : Nat

/-- Calculates the number of rows on the bus given its state -/
def calculate_rows (state : BusState) : Nat :=
  let total_passengers := state.initial + 
    (state.first_stop_board - state.first_stop_leave) + 
    (state.second_stop_board - state.second_stop_leave)
  let total_seats := total_passengers + state.empty_seats
  total_seats / state.seats_per_row

/-- Theorem stating that given the problem conditions, the bus has 23 rows -/
theorem bus_rows_theorem (state : BusState) 
  (h1 : state.initial = 16)
  (h2 : state.first_stop_board = 15)
  (h3 : state.first_stop_leave = 3)
  (h4 : state.second_stop_board = 17)
  (h5 : state.second_stop_leave = 10)
  (h6 : state.empty_seats = 57)
  (h7 : state.seats_per_row = 4) :
  calculate_rows state = 23 := by
  sorry

#eval calculate_rows {
  initial := 16,
  first_stop_board := 15,
  first_stop_leave := 3,
  second_stop_board := 17,
  second_stop_leave := 10,
  empty_seats := 57,
  seats_per_row := 4
}

end bus_rows_theorem_l149_14971


namespace exam_score_problem_l149_14904

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℕ) (total_score : ℕ) (correct_answers : ℕ) :
  total_questions = 60 →
  correct_score = 4 →
  total_score = 120 →
  correct_answers = 36 →
  (total_questions - correct_answers) * (correct_score - (total_score - correct_answers * correct_score) / (total_questions - correct_answers)) = total_questions - correct_answers :=
by sorry

end exam_score_problem_l149_14904


namespace area_of_region_l149_14905

/-- The region defined by the inequality |4x - 20| + |3y - 6| ≤ 4 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |4 * p.1 - 20| + |3 * p.2 - 6| ≤ 4}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem stating that the area of the region is 8/3 -/
theorem area_of_region : area Region = 8/3 := by sorry

end area_of_region_l149_14905


namespace defective_pen_count_l149_14999

theorem defective_pen_count (total_pens : ℕ) (prob_non_defective : ℚ) : 
  total_pens = 12 →
  prob_non_defective = 6/11 →
  (∃ (non_defective : ℕ), 
    (non_defective : ℚ) / total_pens * ((non_defective - 1) : ℚ) / (total_pens - 1) = prob_non_defective ∧
    total_pens - non_defective = 1) :=
by sorry

end defective_pen_count_l149_14999


namespace absolute_value_ratio_l149_14907

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 10*a*b) :
  |((a + b) / (a - b))| = Real.sqrt 6 / 2 := by
  sorry

end absolute_value_ratio_l149_14907


namespace all_statements_imply_negation_l149_14979

theorem all_statements_imply_negation (p q r : Prop) :
  (p ∧ q ∧ ¬r) → (¬p ∨ ¬q ∨ ¬r) ∧
  (¬p ∧ q ∧ r) → (¬p ∨ ¬q ∨ ¬r) ∧
  (p ∧ ¬q ∧ r) → (¬p ∨ ¬q ∨ ¬r) ∧
  (¬p ∧ ¬q ∧ r) → (¬p ∨ ¬q ∨ ¬r) :=
by sorry

#check all_statements_imply_negation

end all_statements_imply_negation_l149_14979


namespace free_younger_son_time_l149_14928

/-- Given a total number of tape strands and cutting rates for Hannah and her son,
    calculate the time needed to cut all strands. -/
def time_to_cut (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ) : ℚ :=
  (total_strands : ℚ) / ((hannah_rate + son_rate) : ℚ)

/-- Theorem stating that it takes 5 minutes to cut 45 strands of tape
    when Hannah cuts 7 strands per minute and her son cuts 2 strands per minute. -/
theorem free_younger_son_time :
  time_to_cut 45 7 2 = 5 := by
  sorry

end free_younger_son_time_l149_14928


namespace wall_width_l149_14940

theorem wall_width (area : ℝ) (height : ℝ) (width : ℝ) :
  area = 8 ∧ height = 4 ∧ area = width * height → width = 2 := by
  sorry

end wall_width_l149_14940


namespace linear_programming_problem_l149_14952

theorem linear_programming_problem (x y a b : ℝ) :
  3 * x - y - 6 ≤ 0 →
  x - y + 2 ≥ 0 →
  x ≥ 0 →
  y ≥ 0 →
  a > 0 →
  b > 0 →
  (∀ x' y', 3 * x' - y' - 6 ≤ 0 → x' - y' + 2 ≥ 0 → x' ≥ 0 → y' ≥ 0 → a * x' + b * y' ≤ 12) →
  a * x + b * y = 12 →
  (2 / a + 3 / b) ≥ 25 / 6 :=
by sorry

end linear_programming_problem_l149_14952


namespace total_amount_calculation_l149_14965

theorem total_amount_calculation (r p q : ℝ) 
  (h1 : r = 2000.0000000000002) 
  (h2 : r = (2/3) * (p + q + r)) : 
  p + q + r = 3000.0000000000003 := by
sorry

end total_amount_calculation_l149_14965


namespace car_initial_speed_l149_14911

/-- Represents a point on the road --/
inductive Point
| A
| B
| C
| D

/-- Represents the speed of the car at different segments --/
structure Speed where
  initial : ℝ
  fromBtoC : ℝ
  fromCtoD : ℝ

/-- Represents the distance between points --/
structure Distance where
  total : ℝ
  AtoB : ℝ
  BtoC : ℝ
  CtoD : ℝ

/-- Represents the travel time between points --/
structure TravelTime where
  BtoC : ℝ
  CtoD : ℝ

/-- The main theorem stating the conditions and the result to be proved --/
theorem car_initial_speed 
  (d : Distance)
  (s : Speed)
  (t : TravelTime)
  (h1 : d.total = 100)
  (h2 : d.total - d.AtoB = 0.5 * s.initial)
  (h3 : s.fromBtoC = s.initial - 10)
  (h4 : s.fromCtoD = s.initial - 20)
  (h5 : d.CtoD = 20)
  (h6 : t.BtoC = t.CtoD + 1/12)
  (h7 : d.BtoC / s.fromBtoC = t.BtoC)
  (h8 : d.CtoD / s.fromCtoD = t.CtoD)
  : s.initial = 100 := by
  sorry


end car_initial_speed_l149_14911


namespace farm_field_solution_l149_14987

/-- Represents the farm field ploughing problem -/
structure FarmField where
  planned_hectares_per_day : ℕ
  actual_hectares_per_day : ℕ
  extra_days : ℕ
  hectares_left : ℕ

/-- Calculates the total area and initial planned days for the farm field -/
def calculate_farm_area_and_days (f : FarmField) : ℕ × ℕ :=
  let initial_days := (f.actual_hectares_per_day * (f.extra_days + 1) + f.hectares_left) / f.planned_hectares_per_day
  let total_area := f.planned_hectares_per_day * initial_days + f.hectares_left
  (total_area, initial_days)

/-- Theorem stating the solution to the farm field problem -/
theorem farm_field_solution (f : FarmField) 
  (h1 : f.planned_hectares_per_day = 160)
  (h2 : f.actual_hectares_per_day = 85)
  (h3 : f.extra_days = 2)
  (h4 : f.hectares_left = 40) :
  calculate_farm_area_and_days f = (520, 3) := by
  sorry

#eval calculate_farm_area_and_days { planned_hectares_per_day := 160, actual_hectares_per_day := 85, extra_days := 2, hectares_left := 40 }

end farm_field_solution_l149_14987


namespace steiner_ellipses_equations_l149_14908

/-- Barycentric coordinates in a triangle -/
structure BarycentricCoord where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Circumscribed Steiner Ellipse equation -/
def circumscribedSteinerEllipse (p : BarycentricCoord) : Prop :=
  p.β * p.γ + p.α * p.γ + p.α * p.β = 0

/-- Inscribed Steiner Ellipse equation -/
def inscribedSteinerEllipse (p : BarycentricCoord) : Prop :=
  2 * p.β * p.γ + 2 * p.α * p.γ + 2 * p.α * p.β = p.α^2 + p.β^2 + p.γ^2

/-- Theorem stating the equations of Steiner ellipses in barycentric coordinates -/
theorem steiner_ellipses_equations (p : BarycentricCoord) :
  (circumscribedSteinerEllipse p ↔ p.β * p.γ + p.α * p.γ + p.α * p.β = 0) ∧
  (inscribedSteinerEllipse p ↔ 2 * p.β * p.γ + 2 * p.α * p.γ + 2 * p.α * p.β = p.α^2 + p.β^2 + p.γ^2) :=
by sorry

end steiner_ellipses_equations_l149_14908


namespace zoo_ticket_cost_zoo_ticket_cost_example_l149_14974

/-- Calculate the total cost of zoo tickets for a group with a discount --/
theorem zoo_ticket_cost (num_children num_adults num_seniors : ℕ)
                        (child_price adult_price senior_price : ℚ)
                        (discount_rate : ℚ) : ℚ :=
  let total_before_discount := num_children * child_price +
                               num_adults * adult_price +
                               num_seniors * senior_price
  let discount_amount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount_amount
  total_after_discount

/-- Prove that the total cost of zoo tickets for the given group is $227.80 --/
theorem zoo_ticket_cost_example : zoo_ticket_cost 6 10 4 10 16 12 (15/100) = 227.8 := by
  sorry

end zoo_ticket_cost_zoo_ticket_cost_example_l149_14974


namespace final_jasmine_concentration_l149_14964

/-- Calculates the final jasmine concentration after adding pure jasmine and water to a solution -/
theorem final_jasmine_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 80)
  (h2 : initial_concentration = 0.1)
  (h3 : added_jasmine = 5)
  (h4 : added_water = 15) :
  let initial_jasmine := initial_volume * initial_concentration
  let final_jasmine := initial_jasmine + added_jasmine
  let final_volume := initial_volume + added_jasmine + added_water
  final_jasmine / final_volume = 0.13 := by
sorry


end final_jasmine_concentration_l149_14964


namespace triangle_properties_l149_14984

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = t.a * Real.cos t.C + (Real.sqrt 3 / 3) * t.a * Real.sin t.C)
  (h2 : t.a = Real.sqrt 7)
  (h3 : t.b * t.c = 6) : 
  t.A = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry

end triangle_properties_l149_14984


namespace circle_equation_l149_14914

theorem circle_equation (x y : ℝ) : 
  (∀ (x₀ y₀ : ℝ), (x₀ = 0 ∧ y₀ = 0) ∨ (x₀ = 4 ∧ y₀ = 0) ∨ (x₀ = -1 ∧ y₀ = 1) → 
    x₀^2 + y₀^2 - 4*x₀ - 6*y₀ = 0) ↔
  x^2 + y^2 - 4*x - 6*y = 0 :=
by sorry

end circle_equation_l149_14914


namespace range_of_function_range_tight_l149_14915

theorem range_of_function (x : ℝ) :
  ∃ (y : ℝ), y = |2 * Real.sin x + 3 * Real.cos x + 4| ∧
  4 - Real.sqrt 13 ≤ y ∧ y ≤ 4 + Real.sqrt 13 :=
by sorry

theorem range_tight :
  ∃ (x₁ x₂ : ℝ), 
    |2 * Real.sin x₁ + 3 * Real.cos x₁ + 4| = 4 - Real.sqrt 13 ∧
    |2 * Real.sin x₂ + 3 * Real.cos x₂ + 4| = 4 + Real.sqrt 13 :=
by sorry

end range_of_function_range_tight_l149_14915


namespace percentage_both_correct_l149_14980

/-- Given a class of students taking a test with two questions, this theorem proves
    the percentage of students who answered both questions correctly. -/
theorem percentage_both_correct
  (p_first : ℝ)  -- Probability of answering the first question correctly
  (p_second : ℝ) -- Probability of answering the second question correctly
  (p_neither : ℝ) -- Probability of answering neither question correctly
  (h1 : p_first = 0.65)  -- 65% answered the first question correctly
  (h2 : p_second = 0.55) -- 55% answered the second question correctly
  (h3 : p_neither = 0.20) -- 20% answered neither question correctly
  : p_first + p_second - (1 - p_neither) = 0.40 := by
  sorry

#check percentage_both_correct

end percentage_both_correct_l149_14980
