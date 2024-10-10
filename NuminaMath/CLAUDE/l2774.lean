import Mathlib

namespace suzanna_bike_ride_l2774_277491

/-- Suzanna's bike ride problem -/
theorem suzanna_bike_ride (rate : ℝ) (time : ℝ) : 
  rate = 3 / 10 → time = 40 → rate * time = 12 := by
  sorry

end suzanna_bike_ride_l2774_277491


namespace ladder_velocity_l2774_277439

theorem ladder_velocity (l a τ : ℝ) (hl : l > 0) (ha : a > 0) (hτ : τ > 0) :
  let α := Real.arcsin (a * τ^2 / (2 * l))
  let v₁ := a * τ
  let v₂ := (a^2 * τ^3) / Real.sqrt (4 * l^2 - a^2 * τ^4)
  v₁ * Real.sin α = v₂ * Real.cos α :=
by sorry

end ladder_velocity_l2774_277439


namespace parabola_chord_intersection_l2774_277479

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = x^2 -/
def parabola (p : Point) : Prop := p.y = p.x^2

/-- Represents the ratio condition AC:CB = 5:2 -/
def ratio_condition (a b c : Point) : Prop :=
  (c.x - a.x) / (b.x - c.x) = 5 / 2

theorem parabola_chord_intersection :
  ∀ (a b c : Point),
    parabola a →
    parabola b →
    c.x = 0 →
    c.y = 20 →
    ratio_condition a b c →
    ((a.x = -5 * Real.sqrt 2 ∧ b.x = 2 * Real.sqrt 2) ∨
     (a.x = 5 * Real.sqrt 2 ∧ b.x = -2 * Real.sqrt 2)) :=
by sorry

end parabola_chord_intersection_l2774_277479


namespace smallest_covering_circle_l2774_277402

-- Define the plane region
def plane_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + 2*y - 4 ≤ 0

-- Define the circle equation
def circle_equation (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem smallest_covering_circle :
  ∃ (a b r : ℝ), 
    (∀ x y, plane_region x y → circle_equation x y a b r) ∧
    (∀ a' b' r', (∀ x y, plane_region x y → circle_equation x y a' b' r') → r' ≥ r) ∧
    a = 2 ∧ b = 1 ∧ r^2 = 5 :=
sorry

end smallest_covering_circle_l2774_277402


namespace area_of_figure_l2774_277493

/-- Given a figure F in the plane of one face of a dihedral angle,
    S is the area of its orthogonal projection onto the other face,
    Q is the area of its orthogonal projection onto the bisector plane.
    This theorem proves that the area T of figure F is equal to (1/2)(√(S² + 8Q²) - S). -/
theorem area_of_figure (S Q : ℝ) (hS : S > 0) (hQ : Q > 0) :
  ∃ T : ℝ, T = (1/2) * (Real.sqrt (S^2 + 8*Q^2) - S) ∧ T > 0 := by
sorry

end area_of_figure_l2774_277493


namespace product_of_sum_and_cube_sum_l2774_277424

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 8) 
  (h2 : a^3 + b^3 = 172) : 
  a * b = 85/6 := by
sorry

end product_of_sum_and_cube_sum_l2774_277424


namespace apple_distribution_equation_l2774_277444

def represents_apple_distribution (x : ℕ) : Prop :=
  (x - 1) % 3 = 0 ∧ (x + 2) % 4 = 0

theorem apple_distribution_equation :
  ∀ x : ℕ, represents_apple_distribution x ↔ (x - 1) / 3 = (x + 2) / 4 :=
by sorry

end apple_distribution_equation_l2774_277444


namespace b_100_mod_81_l2774_277401

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem b_100_mod_81 : b 100 ≡ 38 [ZMOD 81] := by sorry

end b_100_mod_81_l2774_277401


namespace trapezoid_perimeter_and_side_range_l2774_277470

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  top : ℝ
  bottom : ℝ
  side1 : ℝ
  side2 : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.top + t.bottom + t.side1 + t.side2

/-- Theorem stating the relationship between perimeter and side length,
    and the valid range for the variable side length -/
theorem trapezoid_perimeter_and_side_range (x : ℝ) :
  let t := Trapezoid.mk 4 7 12 x
  (perimeter t = x + 23) ∧ (9 < x ∧ x < 15) := by
  sorry


end trapezoid_perimeter_and_side_range_l2774_277470


namespace hemisphere_surface_area_l2774_277499

theorem hemisphere_surface_area (r : Real) : 
  π * r^2 = 3 → 3 * π * r^2 = 9 := by sorry

end hemisphere_surface_area_l2774_277499


namespace rotate_rectangle_is_cylinder_l2774_277497

/-- A rectangle is a 2D shape with four sides and four right angles. -/
structure Rectangle where
  width : ℝ
  height : ℝ
  (positive_dimensions : width > 0 ∧ height > 0)

/-- A cylinder is a 3D shape with two circular bases connected by a curved surface. -/
structure Cylinder where
  radius : ℝ
  height : ℝ
  (positive_dimensions : radius > 0 ∧ height > 0)

/-- The result of rotating a rectangle around one of its sides. -/
def rotateRectangle (r : Rectangle) : Cylinder :=
  sorry

/-- Theorem stating that rotating a rectangle around one of its sides results in a cylinder. -/
theorem rotate_rectangle_is_cylinder (r : Rectangle) :
  ∃ (c : Cylinder), c = rotateRectangle r :=
sorry

end rotate_rectangle_is_cylinder_l2774_277497


namespace largest_decimal_l2774_277453

theorem largest_decimal (a b c d e : ℚ) 
  (ha : a = 0.997) 
  (hb : b = 0.9969) 
  (hc : c = 0.99699) 
  (hd : d = 0.9699) 
  (he : e = 0.999) : 
  e = max a (max b (max c (max d e))) := by
  sorry

end largest_decimal_l2774_277453


namespace disk_arrangement_area_sum_l2774_277490

theorem disk_arrangement_area_sum :
  ∀ (n : ℕ) (r : ℝ) (disk_radius : ℝ),
    n = 15 →
    r = 1 →
    disk_radius = 2 - Real.sqrt 3 →
    (↑n * π * disk_radius^2 : ℝ) = π * (105 - 60 * Real.sqrt 3) ∧
    105 + 60 + 3 = 168 := by
  sorry

end disk_arrangement_area_sum_l2774_277490


namespace inscribed_triangle_area_l2774_277459

/-- A triangle inscribed in a circle with given properties -/
structure InscribedTriangle where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The ratio of the triangle's sides -/
  side_ratio : Fin 3 → ℝ
  /-- The side ratio corresponds to a 3:4:5 triangle -/
  ratio_valid : side_ratio = ![3, 4, 5]
  /-- The radius of the circle is 5 -/
  radius_is_5 : radius = 5

/-- The area of an inscribed triangle with the given properties is 24 -/
theorem inscribed_triangle_area (t : InscribedTriangle) : Real.sqrt (
  (t.side_ratio 0 * t.side_ratio 1 * t.side_ratio 2 * (t.side_ratio 0 + t.side_ratio 1 + t.side_ratio 2)) /
  ((t.side_ratio 0 + t.side_ratio 1) * (t.side_ratio 1 + t.side_ratio 2) * (t.side_ratio 2 + t.side_ratio 0))
) * t.radius ^ 2 = 24 := by
  sorry

end inscribed_triangle_area_l2774_277459


namespace quadratic_function_equality_l2774_277425

theorem quadratic_function_equality (a b c d : ℝ) : 
  (∀ x, (x^2 + a*x + b) = ((2*x + 1)^2 + a*(2*x + 1) + b)) → 
  (∀ x, 4*(x^2 + c*x + d) = ((2*x + 1)^2 + a*(2*x + 1) + b)) → 
  (∀ x, 2*x + a = 2*x + c) → 
  (5^2 + 5*a + b = 30) → 
  (a = 2 ∧ b = -5 ∧ c = 2 ∧ d = -1/2) :=
by sorry

end quadratic_function_equality_l2774_277425


namespace vacant_seats_l2774_277466

def total_seats : ℕ := 600
def filled_percentage : ℚ := 45 / 100

theorem vacant_seats : 
  ⌊(1 - filled_percentage) * total_seats⌋ = 330 := by
  sorry

end vacant_seats_l2774_277466


namespace prism_with_27_edges_has_11_faces_l2774_277426

/-- A prism is a polyhedron with two congruent and parallel faces (bases) connected by lateral faces. -/
structure Prism where
  edges : ℕ
  lateral_faces : ℕ

/-- The number of edges in a prism is three times the number of lateral faces. -/
axiom prism_edge_count (p : Prism) : p.edges = 3 * p.lateral_faces

/-- The total number of faces in a prism is the number of lateral faces plus two (for the bases). -/
def total_faces (p : Prism) : ℕ := p.lateral_faces + 2

/-- Theorem: A prism with 27 edges has 11 faces. -/
theorem prism_with_27_edges_has_11_faces (p : Prism) (h : p.edges = 27) : total_faces p = 11 := by
  sorry


end prism_with_27_edges_has_11_faces_l2774_277426


namespace union_complement_equality_l2774_277408

def U : Set Nat := {0,1,2,3,4,5,6}
def A : Set Nat := {2,4,5}
def B : Set Nat := {0,1,3,5}

theorem union_complement_equality : A ∪ (U \ B) = {2,4,5,6} := by sorry

end union_complement_equality_l2774_277408


namespace square_side_length_l2774_277442

/-- A square with perimeter 24 meters has sides of length 6 meters. -/
theorem square_side_length (s : ℝ) (h1 : s > 0) (h2 : 4 * s = 24) : s = 6 := by
  sorry

end square_side_length_l2774_277442


namespace zero_not_read_in_4006530_l2774_277468

/-- Rule for reading a number -/
def readNumber (n : ℕ) : Bool :=
  sorry

/-- Checks if zero is read out in the number -/
def isZeroReadOut (n : ℕ) : Bool :=
  sorry

theorem zero_not_read_in_4006530 :
  ¬(isZeroReadOut 4006530) ∧ 
  (isZeroReadOut 4650003) ∧ 
  (isZeroReadOut 4650300) ∧ 
  (isZeroReadOut 4006053) := by
  sorry

end zero_not_read_in_4006530_l2774_277468


namespace jeffs_score_l2774_277418

theorem jeffs_score (jeff tim : ℕ) (h1 : jeff = tim + 60) (h2 : (jeff + tim) / 2 = 112) : jeff = 142 := by
  sorry

end jeffs_score_l2774_277418


namespace minimum_value_reciprocal_sum_l2774_277400

theorem minimum_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + 2*n = 1) :
  (1/m + 1/n) ≥ 3 + 2*Real.sqrt 2 ∧ ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m + 2*n = 1 ∧ 1/m + 1/n = 3 + 2*Real.sqrt 2 :=
sorry

end minimum_value_reciprocal_sum_l2774_277400


namespace probability_sum_greater_than_five_l2774_277423

def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

def sample_space : Finset (ℕ × ℕ) := A.product B

def favorable_outcomes : Finset (ℕ × ℕ) := sample_space.filter (fun p => p.1 + p.2 > 5)

theorem probability_sum_greater_than_five :
  (favorable_outcomes.card : ℚ) / sample_space.card = 1 / 3 := by sorry

end probability_sum_greater_than_five_l2774_277423


namespace binary_110110011_to_octal_l2774_277473

def binary_to_octal (b : Nat) : Nat :=
  sorry

theorem binary_110110011_to_octal :
  binary_to_octal 110110011 = 163 := by
  sorry

end binary_110110011_to_octal_l2774_277473


namespace sum_of_two_smallest_numbers_l2774_277495

theorem sum_of_two_smallest_numbers (a b c d : ℝ) : 
  a / b = 3 / 5 ∧ 
  b / c = 5 / 7 ∧ 
  c / d = 7 / 9 ∧ 
  (a + b + c + d) / 4 = 30 →
  a + b = 40 := by
sorry

end sum_of_two_smallest_numbers_l2774_277495


namespace johns_equation_l2774_277446

theorem johns_equation (a b c d e : ℤ) : 
  a = 2 → b = 3 → c = 4 → d = 5 →
  (a - b - c * d + e = a - (b - (c * (d - e)))) →
  e = 8 := by
sorry

end johns_equation_l2774_277446


namespace g_composition_of_3_l2774_277412

def g (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_composition_of_3 : g (g (g (g 3))) = 24 := by
  sorry

end g_composition_of_3_l2774_277412


namespace paper_clip_count_l2774_277440

theorem paper_clip_count (num_boxes : ℕ) (clips_per_box : ℕ) 
  (h1 : num_boxes = 9) (h2 : clips_per_box = 9) : 
  num_boxes * clips_per_box = 81 := by
  sorry

end paper_clip_count_l2774_277440


namespace trip_length_l2774_277485

theorem trip_length (total : ℚ) 
  (h1 : (1 / 4 : ℚ) * total + 25 + (1 / 6 : ℚ) * total = total) :
  total = 300 / 7 := by
  sorry

end trip_length_l2774_277485


namespace triangle_area_ratio_l2774_277472

theorem triangle_area_ratio (BD DC : ℝ) (area_ABD : ℝ) (area_ADC : ℝ) :
  BD / DC = 5 / 2 →
  area_ABD = 40 →
  area_ADC = area_ABD * (DC / BD) →
  area_ADC = 16 :=
by sorry

end triangle_area_ratio_l2774_277472


namespace weather_probability_l2774_277435

theorem weather_probability (p_rain p_cloudy : ℝ) 
  (h_rain : p_rain = 0.45)
  (h_cloudy : p_cloudy = 0.20)
  (h_nonneg_rain : 0 ≤ p_rain)
  (h_nonneg_cloudy : 0 ≤ p_cloudy)
  (h_sum_le_one : p_rain + p_cloudy ≤ 1) :
  1 - p_rain - p_cloudy = 0.35 := by
  sorry

end weather_probability_l2774_277435


namespace nova_monthly_donation_l2774_277456

/-- Nova's monthly donation to charity -/
def monthly_donation : ℕ := 1707

/-- Nova's total annual donation to charity -/
def annual_donation : ℕ := 20484

/-- Number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: Nova's monthly donation is $1,707 -/
theorem nova_monthly_donation :
  monthly_donation = annual_donation / months_in_year :=
by sorry

end nova_monthly_donation_l2774_277456


namespace power_of_64_two_thirds_l2774_277413

theorem power_of_64_two_thirds : (64 : ℝ) ^ (2/3) = 16 := by
  sorry

end power_of_64_two_thirds_l2774_277413


namespace isosceles_triangle_l2774_277455

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the condition c = 2a cos B
def condition (t : Triangle) : Prop :=
  t.c = 2 * t.a * Real.cos t.angleB

-- State the theorem
theorem isosceles_triangle (t : Triangle) (h : condition t) : t.a = t.b := by
  sorry

end isosceles_triangle_l2774_277455


namespace sphere_volume_increase_l2774_277404

/-- Proves that when the surface area of a sphere is increased to 4 times its original size,
    its volume is increased to 8 times the original. -/
theorem sphere_volume_increase (r : ℝ) (S V : ℝ → ℝ) 
    (hS : ∃ k : ℝ, ∀ x, S x = k * x^2)  -- Surface area is proportional to radius squared
    (hV : ∃ c : ℝ, ∀ x, V x = c * x^3)  -- Volume is proportional to radius cubed
    (hS_increase : S (2 * r) = 4 * S r) : -- Surface area is increased 4 times
  V (2 * r) = 8 * V r := by
sorry


end sphere_volume_increase_l2774_277404


namespace transmitter_find_probability_l2774_277427

/-- Represents a license plate format for government vehicles in Kerrania -/
structure LicensePlate :=
  (first_two : Fin 100)
  (second : Fin 10)
  (last_two : Fin 100)
  (letters : Fin 3 × Fin 3)

/-- Conditions for a valid government license plate -/
def is_valid_plate (plate : LicensePlate) : Prop :=
  plate.first_two = 79 ∧
  (plate.second = 3 ∨ plate.second = 5) ∧
  (plate.last_two / 10 = plate.last_two % 10)

/-- Number of vehicles police can inspect in 3 hours -/
def inspected_vehicles : ℕ := 18

/-- Total number of possible valid license plates -/
def total_valid_plates : ℕ := 180

/-- Probability of finding the transmitter within 3 hours -/
def find_probability : ℚ := 1 / 10

theorem transmitter_find_probability :
  (inspected_vehicles : ℚ) / total_valid_plates = find_probability :=
sorry

end transmitter_find_probability_l2774_277427


namespace inscribed_rectangle_area_coefficient_l2774_277460

/-- Triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rectangle inscribed in a triangle -/
structure InscribedRectangle (t : Triangle) where
  area : ℝ → ℝ

/-- The theorem stating the value of β in the area formula of the inscribed rectangle -/
theorem inscribed_rectangle_area_coefficient (t : Triangle) 
  (r : InscribedRectangle t) : 
  t.a = 12 → t.b = 25 → t.c = 17 → 
  (∃ α β : ℝ, ∀ ω, r.area ω = α * ω - β * ω^2) → 
  (∃ β : ℝ, β = 36 / 125) := by
  sorry

end inscribed_rectangle_area_coefficient_l2774_277460


namespace complex_equation_solution_l2774_277416

theorem complex_equation_solution (z : ℂ) : 
  (1 + 2*I)*z = 4 + 3*I → z = 2 - I := by
  sorry

end complex_equation_solution_l2774_277416


namespace triangle_inequality_l2774_277407

theorem triangle_inequality (a b c : ℝ) (C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hC : 0 < C ∧ C < π) :
  c ≥ (a + b) * Real.sin (C / 2) := by
  sorry

end triangle_inequality_l2774_277407


namespace circle_center_l2774_277441

/-- Given a circle with diameter endpoints (3, -3) and (13, 17), its center is (8, 7) -/
theorem circle_center (Q : Set (ℝ × ℝ)) (p₁ p₂ : ℝ × ℝ) (h₁ : p₁ = (3, -3)) (h₂ : p₂ = (13, 17)) 
    (h₃ : ∀ x ∈ Q, ∃ y ∈ Q, (x.1 - y.1)^2 + (x.2 - y.2)^2 = (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) :
  ∃ c : ℝ × ℝ, c = (8, 7) ∧ ∀ x ∈ Q, (x.1 - c.1)^2 + (x.2 - c.2)^2 = ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) / 4 :=
by
  sorry

end circle_center_l2774_277441


namespace remainder_3_1000_mod_7_l2774_277406

theorem remainder_3_1000_mod_7 : 3^1000 % 7 = 4 := by
  sorry

end remainder_3_1000_mod_7_l2774_277406


namespace derivative_f_at_half_l2774_277465

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x + 1

-- State the theorem
theorem derivative_f_at_half : 
  deriv f (1/2) = -2 :=
sorry

end derivative_f_at_half_l2774_277465


namespace symmetric_circle_l2774_277419

/-- Given a circle and a line of symmetry, find the equation of the symmetric circle -/
theorem symmetric_circle (x y : ℝ) : 
  -- Original circle
  ((x - 2)^2 + (y - 3)^2 = 1) →
  -- Line of symmetry
  (x + y - 1 = 0) →
  -- Symmetric circle
  ((x + 2)^2 + (y + 1)^2 = 1) :=
by
  sorry

end symmetric_circle_l2774_277419


namespace total_marbles_after_exchanges_l2774_277420

def initial_green : ℕ := 32
def initial_violet : ℕ := 38
def initial_blue : ℕ := 46

def mike_takes_green : ℕ := 23
def mike_gives_red : ℕ := 15
def alice_takes_violet : ℕ := 15
def alice_gives_yellow : ℕ := 20
def bob_takes_blue : ℕ := 31
def bob_gives_white : ℕ := 12

def mike_returns_green : ℕ := 10
def mike_takes_red : ℕ := 7
def alice_returns_violet : ℕ := 8
def alice_takes_yellow : ℕ := 9
def bob_returns_blue : ℕ := 17
def bob_takes_white : ℕ := 5

def final_green : ℕ := initial_green - mike_takes_green + mike_returns_green
def final_violet : ℕ := initial_violet - alice_takes_violet + alice_returns_violet
def final_blue : ℕ := initial_blue - bob_takes_blue + bob_returns_blue
def final_red : ℕ := mike_gives_red - mike_takes_red
def final_yellow : ℕ := alice_gives_yellow - alice_takes_yellow
def final_white : ℕ := bob_gives_white - bob_takes_white

theorem total_marbles_after_exchanges :
  final_green + final_violet + final_blue + final_red + final_yellow + final_white = 108 :=
by sorry

end total_marbles_after_exchanges_l2774_277420


namespace mark_reading_time_l2774_277483

/-- Mark's daily reading time in hours -/
def daily_reading_time : ℕ := 2

/-- Number of days in a week -/
def days_per_week : ℕ := 7

/-- Mark's planned increase in weekly reading time in hours -/
def weekly_increase : ℕ := 4

/-- Mark's desired weekly reading time in hours -/
def desired_weekly_reading_time : ℕ := daily_reading_time * days_per_week + weekly_increase

theorem mark_reading_time :
  desired_weekly_reading_time = 18 := by
  sorry

end mark_reading_time_l2774_277483


namespace circle_center_correct_l2774_277462

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 + 6*y - 16 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (-2, -3)

/-- Theorem: The center of the circle with equation x^2 + 4x + y^2 + 6y - 16 = 0 is (-2, -3) -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = 29 :=
by sorry

end circle_center_correct_l2774_277462


namespace smallest_valid_number_l2774_277417

def is_valid_number (N : ℕ) : Prop :=
  ∃ X : ℕ, 
    X > 0 ∧
    (N - 12) % 8 = 0 ∧
    (N - 12) % 12 = 0 ∧
    (N - 12) % 24 = 0 ∧
    (N - 12) % X = 0 ∧
    (N - 12) / Nat.lcm 24 X = 276

theorem smallest_valid_number : 
  is_valid_number 6636 ∧ ∀ n < 6636, ¬ is_valid_number n :=
sorry

end smallest_valid_number_l2774_277417


namespace point_difference_on_line_l2774_277492

/-- Given two points (m, n) and (m + v, n + 18) on the line x = (y / 6) - (2 / 5),
    prove that v = 3 -/
theorem point_difference_on_line (m n : ℝ) :
  (m = n / 6 - 2 / 5) →
  (m + 3 = (n + 18) / 6 - 2 / 5) := by
sorry

end point_difference_on_line_l2774_277492


namespace platform_length_problem_solution_l2774_277477

/-- Calculates the length of a platform given train parameters --/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * crossing_time
  total_distance - train_length

/-- The length of the platform is 208.8 meters --/
theorem problem_solution : 
  (platform_length 180 70 20) = 208.8 := by
  sorry

end platform_length_problem_solution_l2774_277477


namespace lucas_cleaning_days_l2774_277415

/-- Calculates the number of days Lucas took to clean windows -/
def days_to_clean (floors : ℕ) (windows_per_floor : ℕ) (payment_per_window : ℕ) 
                  (deduction_per_period : ℕ) (days_per_period : ℕ) (final_payment : ℕ) : ℕ :=
  let total_windows := floors * windows_per_floor
  let total_payment := total_windows * payment_per_window
  let deduction := total_payment - final_payment
  let periods := deduction / deduction_per_period
  periods * days_per_period

/-- Theorem stating that Lucas took 6 days to clean all windows -/
theorem lucas_cleaning_days : 
  days_to_clean 3 3 2 1 3 16 = 6 := by sorry

end lucas_cleaning_days_l2774_277415


namespace crayons_left_l2774_277496

theorem crayons_left (initial : ℝ) (taken : ℝ) (left : ℝ) : 
  initial = 7.5 → taken = 2.25 → left = initial - taken → left = 5.25 := by
  sorry

end crayons_left_l2774_277496


namespace floor_times_self_eq_120_l2774_277476

theorem floor_times_self_eq_120 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 120 ∧ x = 120 / 11 :=
by sorry

end floor_times_self_eq_120_l2774_277476


namespace profit_maximization_l2774_277450

noncomputable def g (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 30 then
    -5 * x^2 + 420 * x - 3
  else if 30 < x ∧ x ≤ 110 then
    -2 * x - 20000 / (x + 10) + 597
  else
    0

theorem profit_maximization :
  ∃ (x : ℝ), 0 < x ∧ x ≤ 110 ∧
  g x = 9320 ∧
  ∀ (y : ℝ), 0 < y ∧ y ≤ 110 → g y ≤ g x :=
by sorry

end profit_maximization_l2774_277450


namespace unique_solution_star_l2774_277481

/-- The ⋆ operation -/
def star (x y : ℝ) : ℝ := 5*x - 2*y + 3*x*y

/-- Theorem stating that 2 ⋆ y = 10 has a unique solution y = 0 -/
theorem unique_solution_star :
  ∃! y : ℝ, star 2 y = 10 :=
sorry

end unique_solution_star_l2774_277481


namespace hyperbola_vertices_distance_l2774_277411

/-- The distance between vertices of a hyperbola -/
def distance_between_vertices (a b : ℝ) : ℝ := 2 * a

/-- The equation of the hyperbola -/
def is_hyperbola (x y a b : ℝ) : Prop :=
  x^2 / (a^2) - y^2 / (b^2) = 1

theorem hyperbola_vertices_distance :
  ∀ x y : ℝ, is_hyperbola x y 4 2 → distance_between_vertices 4 2 = 8 := by
  sorry

end hyperbola_vertices_distance_l2774_277411


namespace diophantine_equation_solution_exists_l2774_277489

theorem diophantine_equation_solution_exists :
  ∃ (x y z : ℕ+), 
    (z = Nat.gcd x y) ∧ 
    (x + y^2 + z^3 = x * y * z) := by
  sorry

end diophantine_equation_solution_exists_l2774_277489


namespace touching_circles_perimeter_l2774_277429

/-- Given a circle with center O and radius R, and two smaller circles with centers O₁ and O₂
    that touch each other and internally touch the larger circle,
    the perimeter of triangle OO₁O₂ is 2R. -/
theorem touching_circles_perimeter (O O₁ O₂ : ℝ × ℝ) (R : ℝ) :
  (∃ R₁ R₂ : ℝ, 
    R₁ > 0 ∧ R₂ > 0 ∧
    dist O O₁ = R - R₁ ∧
    dist O O₂ = R - R₂ ∧
    dist O₁ O₂ = R₁ + R₂) →
  dist O O₁ + dist O O₂ + dist O₁ O₂ = 2 * R :=
by sorry


end touching_circles_perimeter_l2774_277429


namespace optimal_avocado_buying_strategy_l2774_277434

/-- Represents the optimal buying strategy for avocados -/
theorem optimal_avocado_buying_strategy 
  (recipe_need : ℕ) 
  (initial_count : ℕ) 
  (price_less_than_10 : ℝ) 
  (price_10_or_more : ℝ) 
  (h1 : recipe_need = 3) 
  (h2 : initial_count = 5) 
  (h3 : price_10_or_more < price_less_than_10) : 
  let additional_buy := 5
  let total_count := initial_count + additional_buy
  let total_cost := additional_buy * price_10_or_more
  (∀ n : ℕ, 
    let alt_total_count := initial_count + n
    let alt_total_cost := if alt_total_count < 10 then n * price_less_than_10 else n * price_10_or_more
    (alt_total_count ≥ recipe_need → total_cost ≤ alt_total_cost) ∧ 
    (alt_total_cost = total_cost → total_count ≥ alt_total_count)) :=
by sorry

end optimal_avocado_buying_strategy_l2774_277434


namespace unique_value_2n_plus_m_l2774_277409

theorem unique_value_2n_plus_m : ∃! v : ℤ, ∀ n m : ℤ,
  (3 * n - m < 5) →
  (n + m > 26) →
  (3 * m - 2 * n < 46) →
  (2 * n + m = v) := by
  sorry

end unique_value_2n_plus_m_l2774_277409


namespace gift_shop_pricing_l2774_277475

theorem gift_shop_pricing (p : ℝ) (hp : p > 0) : 
  0.7 * (1.1 * p) = 0.77 * p := by sorry

end gift_shop_pricing_l2774_277475


namespace article_pricing_l2774_277469

/-- The value of x when the cost price of 20 articles equals the selling price of x articles with a 25% profit -/
theorem article_pricing (C : ℝ) (x : ℝ) (h1 : C > 0) :
  20 * C = x * (C * (1 + 0.25)) → x = 16 := by
  sorry

end article_pricing_l2774_277469


namespace max_daily_sales_revenue_l2774_277452

def f (t : ℕ) : ℚ :=
  if t < 15 then (1/3) * t + 8 else -(1/3) * t + 18

def g (t : ℕ) : ℚ := -t + 30

def W (t : ℕ) : ℚ := (f t) * (g t)

theorem max_daily_sales_revenue (t : ℕ) (h : 0 < t ∧ t ≤ 30) : 
  W t ≤ 243 ∧ ∃ t₀ : ℕ, 0 < t₀ ∧ t₀ ≤ 30 ∧ W t₀ = 243 :=
sorry

end max_daily_sales_revenue_l2774_277452


namespace percent_juniors_in_sports_l2774_277474

theorem percent_juniors_in_sports (total_students : ℕ) (percent_juniors : ℚ) (juniors_in_sports : ℕ) :
  total_students = 500 →
  percent_juniors = 40 / 100 →
  juniors_in_sports = 140 →
  (juniors_in_sports : ℚ) / (percent_juniors * total_students) * 100 = 70 := by
  sorry


end percent_juniors_in_sports_l2774_277474


namespace electronic_product_pricing_l2774_277451

/-- The marked price of an electronic product -/
def marked_price : ℝ := 28

/-- The cost price of the electronic product -/
def cost_price : ℝ := 21

/-- The selling price ratio (90% of marked price) -/
def selling_price_ratio : ℝ := 0.9

/-- The profit ratio (20%) -/
def profit_ratio : ℝ := 0.2

theorem electronic_product_pricing :
  selling_price_ratio * marked_price - cost_price = profit_ratio * cost_price :=
by sorry

end electronic_product_pricing_l2774_277451


namespace quadratic_inequality_implies_a_bound_l2774_277463

theorem quadratic_inequality_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x - a > 0) → a < -1 := by
  sorry

end quadratic_inequality_implies_a_bound_l2774_277463


namespace rectangle_area_plus_perimeter_l2774_277447

/-- Represents a rectangle with positive integer side lengths -/
structure Rectangle where
  length : ℕ+
  width : ℕ+

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length.val * r.width.val

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length.val + r.width.val)

/-- Predicate to check if a number can be expressed as the sum of area and perimeter -/
def canBeExpressedAsAreaPlusPerimeter (n : ℕ) : Prop :=
  ∃ r : Rectangle, area r + perimeter r = n

theorem rectangle_area_plus_perimeter :
  (canBeExpressedAsAreaPlusPerimeter 100) ∧
  (canBeExpressedAsAreaPlusPerimeter 104) ∧
  (canBeExpressedAsAreaPlusPerimeter 106) ∧
  (canBeExpressedAsAreaPlusPerimeter 108) ∧
  ¬(canBeExpressedAsAreaPlusPerimeter 102) := by sorry

end rectangle_area_plus_perimeter_l2774_277447


namespace original_number_is_200_l2774_277414

theorem original_number_is_200 : 
  ∃ x : ℝ, (x - 25 = 0.75 * x + 25) ∧ x = 200 := by
  sorry

end original_number_is_200_l2774_277414


namespace function_inequality_l2774_277484

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 1) * deriv f x ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end function_inequality_l2774_277484


namespace inequality_proof_l2774_277428

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1) / (y + 1) + (y + 1) / (z + 1) + (z + 1) / (x + 1) ≤ x / y + y / z + z / x :=
by sorry

end inequality_proof_l2774_277428


namespace arithmetic_sequence_middle_term_l2774_277445

/-- Three real numbers form an arithmetic sequence if the middle term is the arithmetic mean of the other two terms. -/
def is_arithmetic_sequence (a b c : ℝ) : Prop := b = (a + c) / 2

/-- If 2, m, 6 form an arithmetic sequence, then m = 4 -/
theorem arithmetic_sequence_middle_term : 
  ∀ m : ℝ, is_arithmetic_sequence 2 m 6 → m = 4 := by
  sorry

end arithmetic_sequence_middle_term_l2774_277445


namespace min_distance_to_circle_l2774_277448

theorem min_distance_to_circle (x y : ℝ) :
  x^2 + y^2 - 4*x + 6*y + 4 = 0 →
  ∃ (min : ℝ), min = Real.sqrt 13 - 3 ∧
    ∀ (a b : ℝ), a^2 + b^2 - 4*a + 6*b + 4 = 0 →
      Real.sqrt (a^2 + b^2) ≥ min :=
by sorry

end min_distance_to_circle_l2774_277448


namespace granola_net_profit_l2774_277437

/-- Calculates the net profit from selling granola bags --/
theorem granola_net_profit
  (cost_per_bag : ℝ)
  (total_bags : ℕ)
  (full_price : ℝ)
  (discounted_price : ℝ)
  (bags_sold_full : ℕ)
  (bags_sold_discounted : ℕ)
  (h1 : cost_per_bag = 3)
  (h2 : total_bags = 20)
  (h3 : full_price = 6)
  (h4 : discounted_price = 4)
  (h5 : bags_sold_full = 15)
  (h6 : bags_sold_discounted = 5)
  (h7 : bags_sold_full + bags_sold_discounted = total_bags) :
  (full_price * bags_sold_full + discounted_price * bags_sold_discounted) - (cost_per_bag * total_bags) = 50 := by
  sorry

#check granola_net_profit

end granola_net_profit_l2774_277437


namespace minimum_score_for_raised_average_l2774_277422

def current_scores : List ℝ := [88, 92, 75, 85, 80]
def raise_average : ℝ := 5

theorem minimum_score_for_raised_average 
  (scores : List ℝ) 
  (raise : ℝ) 
  (h1 : scores = current_scores) 
  (h2 : raise = raise_average) : 
  (scores.sum + (scores.length + 1) * (scores.sum / scores.length + raise) - scores.sum) = 114 :=
sorry

end minimum_score_for_raised_average_l2774_277422


namespace merchant_savings_l2774_277478

def initial_order : ℝ := 15000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.1, 0.3, 0.2]
def option2_discounts : List ℝ := [0.25, 0.15, 0.05]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

theorem merchant_savings :
  apply_successive_discounts initial_order option2_discounts -
  apply_successive_discounts initial_order option1_discounts = 1524.38 := by
  sorry

end merchant_savings_l2774_277478


namespace circle_center_l2774_277405

theorem circle_center (x y : ℝ) : 
  4 * x^2 - 16 * x + 4 * y^2 + 8 * y - 12 = 0 → 
  ∃ (h k : ℝ), h = 2 ∧ k = -1 ∧ (x - h)^2 + (y - k)^2 = 8 :=
by sorry

end circle_center_l2774_277405


namespace smallest_m_for_integral_solutions_l2774_277461

theorem smallest_m_for_integral_solutions : 
  let f (m : ℕ) (x : ℤ) := 12 * x^2 - m * x + 360
  ∃ (m₀ : ℕ), m₀ > 0 ∧ 
    (∃ (x : ℤ), f m₀ x = 0) ∧ 
    (∀ (m : ℕ), 0 < m ∧ m < m₀ → ∀ (x : ℤ), f m x ≠ 0) ∧
    m₀ = 132 :=
by sorry

end smallest_m_for_integral_solutions_l2774_277461


namespace inequality_proof_l2774_277432

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * Real.sqrt (a * b)) / (Real.sqrt a + Real.sqrt b) ≤ (a * b) ^ (1/4) := by
  sorry

end inequality_proof_l2774_277432


namespace valid_three_digit_numbers_l2774_277480

/-- The count of valid three-digit numbers -/
def valid_count : ℕ := 738

/-- The total count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with exactly two non-adjacent identical digits -/
def count_two_same_not_adjacent : ℕ := 81

/-- The count of three-digit numbers with identical first and last digits -/
def count_first_last_same : ℕ := 81

/-- Theorem stating the count of valid three-digit numbers -/
theorem valid_three_digit_numbers :
  valid_count = total_three_digit_numbers - count_two_same_not_adjacent - count_first_last_same :=
by sorry

end valid_three_digit_numbers_l2774_277480


namespace lesser_number_l2774_277421

theorem lesser_number (x y : ℝ) (sum : x + y = 60) (diff : x - y = 8) : 
  min x y = 26 := by
sorry

end lesser_number_l2774_277421


namespace product_real_part_l2774_277464

/-- Given two complex numbers a and b with magnitudes 3 and 5 respectively,
    prove that the positive real part of their product is 6√6. -/
theorem product_real_part (a b : ℂ) (ha : Complex.abs a = 3) (hb : Complex.abs b = 5) :
  (a * b).re = 6 * Real.sqrt 6 := by
  sorry

end product_real_part_l2774_277464


namespace random_variables_comparison_l2774_277486

-- Define the random variables ξ and η
def ξ (a b c : ℝ) : ℝ → ℝ := sorry
def η (a b c : ℝ) : ℝ → ℝ := sorry

-- Define the probability measure
def P : Set ℝ → ℝ := sorry

-- Define expected value
def E (X : ℝ → ℝ) : ℝ := sorry

-- Define variance
def D (X : ℝ → ℝ) : ℝ := sorry

theorem random_variables_comparison (a b c : ℝ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  (E (ξ a b c) = E (η a b c)) ∧ (D (ξ a b c) > D (η a b c)) := by
  sorry

end random_variables_comparison_l2774_277486


namespace identity_proof_l2774_277494

theorem identity_proof (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end identity_proof_l2774_277494


namespace karlson_candies_theorem_l2774_277403

/-- The number of ones initially on the board -/
def initial_ones : ℕ := 29

/-- The number of minutes the process continues -/
def total_minutes : ℕ := 29

/-- Calculates the number of edges in a complete graph with n vertices -/
def complete_graph_edges (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- The maximum number of candies Karlson could eat -/
def max_candies : ℕ := complete_graph_edges initial_ones

theorem karlson_candies_theorem :
  max_candies = 406 :=
sorry

end karlson_candies_theorem_l2774_277403


namespace novel_sales_ratio_l2774_277443

theorem novel_sales_ratio : 
  let total_copies : ℕ := 440000
  let paperback_copies : ℕ := 363600
  let initial_hardback : ℕ := 36000
  let hardback_copies := total_copies - paperback_copies
  let later_hardback := hardback_copies - initial_hardback
  let later_paperback := paperback_copies
  (later_paperback : ℚ) / (later_hardback : ℚ) = 9 / 1 :=
by sorry

end novel_sales_ratio_l2774_277443


namespace analogous_property_is_about_surfaces_l2774_277454

/-- Represents a geometric property in plane geometry -/
structure PlaneProperty where
  description : String

/-- Represents a geometric property in solid geometry -/
structure SolidProperty where
  description : String

/-- Represents the analogy between plane and solid geometry properties -/
def analogy (plane : PlaneProperty) : SolidProperty :=
  sorry

/-- The plane geometry property about equilateral triangles -/
def triangle_property : PlaneProperty :=
  { description := "The sum of distances from any point inside an equilateral triangle to its three sides is constant" }

/-- Theorem stating that the analogous property in solid geometry is about surfaces -/
theorem analogous_property_is_about_surfaces :
  ∃ (surface_prop : SolidProperty),
    (analogy triangle_property).description = "A property about surfaces" :=
  sorry

end analogous_property_is_about_surfaces_l2774_277454


namespace solution_set1_correct_solution_set2_correct_l2774_277498

-- Define the solution set for the first inequality
def solutionSet1 : Set ℝ := {x | x ≥ 1 ∨ x < 0}

-- Define the solution set for the second inequality based on the value of a
def solutionSet2 (a : ℝ) : Set ℝ :=
  if a = 0 then
    {x | x > -1}
  else if a > 0 then
    {x | -1 < x ∧ x < 1/a}
  else if a < -1 then
    {x | x < -1 ∨ x > 1/a}
  else if a = -1 then
    {x | x ≠ -1}
  else
    {x | x < 1/a ∨ x > -1}

-- Theorem for the first inequality
theorem solution_set1_correct :
  ∀ x : ℝ, x ∈ solutionSet1 ↔ (x - 1) / x ≥ 0 ∧ x ≠ 0 :=
sorry

-- Theorem for the second inequality
theorem solution_set2_correct :
  ∀ a x : ℝ, x ∈ solutionSet2 a ↔ a * x^2 + (a - 1) * x - 1 < 0 :=
sorry

end solution_set1_correct_solution_set2_correct_l2774_277498


namespace ln_inequality_solution_set_l2774_277436

theorem ln_inequality_solution_set :
  {x : ℝ | Real.log (2 * x - 1) < 0} = Set.Ioo (1/2 : ℝ) 1 := by sorry

end ln_inequality_solution_set_l2774_277436


namespace solution_sum_of_squares_l2774_277471

theorem solution_sum_of_squares (x y : ℝ) : 
  x * y = 8 → 
  x^2 * y + x * y^2 + 2*x + 2*y = 108 → 
  x^2 + y^2 = 100.64 := by
sorry

end solution_sum_of_squares_l2774_277471


namespace polynomial_sum_l2774_277449

theorem polynomial_sum (f g : ℝ → ℝ) :
  (∀ x, f x + g x = 3 * x - x^2) →
  (∀ x, f x = x^2 - 4 * x + 3) →
  (∀ x, g x = -2 * x^2 + 7 * x - 3) := by
  sorry

end polynomial_sum_l2774_277449


namespace applicants_with_experience_and_degree_l2774_277410

theorem applicants_with_experience_and_degree 
  (total : ℕ) 
  (experienced : ℕ) 
  (degreed : ℕ) 
  (inexperienced_no_degree : ℕ) 
  (h1 : total = 30)
  (h2 : experienced = 10)
  (h3 : degreed = 18)
  (h4 : inexperienced_no_degree = 3) :
  total - (experienced + degreed - (total - inexperienced_no_degree)) = 1 := by
sorry

end applicants_with_experience_and_degree_l2774_277410


namespace shortest_side_in_triangle_l2774_277458

theorem shortest_side_in_triangle (A B C : Real) (a b c : Real) :
  B = 45 * π / 180 →
  C = 60 * π / 180 →
  c = 1 →
  b = Real.sqrt 6 / 3 →
  b < a ∧ b < c :=
by
  sorry

end shortest_side_in_triangle_l2774_277458


namespace tan_alpha_value_l2774_277482

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 2) : Real.tan α = 1/3 := by
  sorry

end tan_alpha_value_l2774_277482


namespace imaginary_unit_power_l2774_277430

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2016 = 1 := by
  sorry

end imaginary_unit_power_l2774_277430


namespace tangent_through_origin_l2774_277431

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x

theorem tangent_through_origin (x₀ : ℝ) (h₁ : x₀ > 0) :
  (∃ k : ℝ, k * x₀ = f x₀ ∧ ∀ x : ℝ, f x₀ + k * (x - x₀) = k * x) →
  x₀ = Real.exp 1 := by
  sorry

end tangent_through_origin_l2774_277431


namespace tirzah_handbags_l2774_277457

/-- The number of handbags Tirzah has -/
def num_handbags : ℕ := 24

/-- The total number of purses Tirzah has -/
def total_purses : ℕ := 26

/-- The fraction of fake purses -/
def fake_purses_fraction : ℚ := 1/2

/-- The fraction of fake handbags -/
def fake_handbags_fraction : ℚ := 1/4

/-- The total number of authentic items (purses and handbags) -/
def total_authentic : ℕ := 31

theorem tirzah_handbags :
  num_handbags = 24 ∧
  total_purses = 26 ∧
  fake_purses_fraction = 1/2 ∧
  fake_handbags_fraction = 1/4 ∧
  total_authentic = 31 ∧
  (total_purses : ℚ) * (1 - fake_purses_fraction) + (num_handbags : ℚ) * (1 - fake_handbags_fraction) = total_authentic := by
  sorry

end tirzah_handbags_l2774_277457


namespace water_percentage_in_first_liquid_l2774_277488

/-- Given two liquids in a glass, prove the percentage of water in the first liquid --/
theorem water_percentage_in_first_liquid 
  (water_percent_second : ℝ) 
  (parts_first : ℝ) 
  (parts_second : ℝ) 
  (water_percent_mixture : ℝ) : 
  water_percent_second = 0.35 →
  parts_first = 10 →
  parts_second = 4 →
  water_percent_mixture = 0.24285714285714285 →
  ∃ (water_percent_first : ℝ), 
    water_percent_first * parts_first + water_percent_second * parts_second = 
    water_percent_mixture * (parts_first + parts_second) ∧
    water_percent_first = 0.2 := by
  sorry

end water_percentage_in_first_liquid_l2774_277488


namespace find_divisor_l2774_277467

theorem find_divisor (n m d : ℕ) (h1 : n - 7 = m) (h2 : m % d = 0) (h3 : ∀ k < 7, (n - k) % d ≠ 0) : d = 7 := by
  sorry

end find_divisor_l2774_277467


namespace problem_statement_l2774_277433

theorem problem_statement (x y : ℝ) (hx : x = 7) (hy : y = -2) :
  (x - 2*y)^y = 1/121 := by sorry

end problem_statement_l2774_277433


namespace greta_hourly_wage_l2774_277487

/-- Greta's hourly wage in dollars -/
def greta_wage : ℝ := 12

/-- The number of hours Greta worked -/
def greta_hours : ℕ := 40

/-- Lisa's hourly wage in dollars -/
def lisa_wage : ℝ := 15

/-- The number of hours Lisa would need to work to equal Greta's earnings -/
def lisa_hours : ℕ := 32

theorem greta_hourly_wage :
  greta_wage * greta_hours = lisa_wage * lisa_hours := by sorry

end greta_hourly_wage_l2774_277487


namespace isosceles_triangle_quadratic_roots_l2774_277438

theorem isosceles_triangle_quadratic_roots (a b k : ℝ) : 
  (∃ c : ℝ, c = 4 ∧ 
   (a = b ∧ (a + b = c ∨ a + c = b ∨ b + c = a)) ∧
   a^2 - 12*a + k + 2 = 0 ∧
   b^2 - 12*b + k + 2 = 0) →
  k = 34 ∨ k = 30 := by
sorry

end isosceles_triangle_quadratic_roots_l2774_277438
