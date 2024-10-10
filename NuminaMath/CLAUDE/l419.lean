import Mathlib

namespace fertilizer_growth_rate_l419_41968

theorem fertilizer_growth_rate 
  (april_output : ℝ) 
  (may_decrease : ℝ) 
  (july_output : ℝ) 
  (h1 : april_output = 500)
  (h2 : may_decrease = 0.2)
  (h3 : july_output = 576) :
  ∃ (x : ℝ), 
    april_output * (1 - may_decrease) * (1 + x)^2 = july_output ∧ 
    x = 0.2 :=
sorry

end fertilizer_growth_rate_l419_41968


namespace largest_prime_factor_of_sum_of_divisors_450_l419_41988

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_450 :
  let M := sum_of_divisors 450
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ M ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ M → q ≤ p ∧ p = 31 :=
sorry

end largest_prime_factor_of_sum_of_divisors_450_l419_41988


namespace line_perp_parallel_implies_plane_perp_l419_41918

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_plane_perp 
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → planePerp α β :=
sorry

end line_perp_parallel_implies_plane_perp_l419_41918


namespace doug_lost_marbles_l419_41919

theorem doug_lost_marbles (d : ℕ) (l : ℕ) : 
  (d + 22 = d - l + 30) → l = 8 := by
  sorry

end doug_lost_marbles_l419_41919


namespace rahul_mary_age_difference_l419_41960

/-- 
Given:
- Mary's current age is 10 years
- In 20 years, Rahul will be twice as old as Mary

Prove that Rahul is currently 30 years older than Mary
-/
theorem rahul_mary_age_difference :
  ∀ (rahul_age mary_age : ℕ),
    mary_age = 10 →
    rahul_age + 20 = 2 * (mary_age + 20) →
    rahul_age - mary_age = 30 :=
by sorry

end rahul_mary_age_difference_l419_41960


namespace frame_diameter_l419_41975

/-- Given two circular frames X and Y, where X has a diameter of 16 cm and Y covers 0.5625 of X's area, prove that Y's diameter is 12 cm. -/
theorem frame_diameter (dX : ℝ) (coverage : ℝ) (dY : ℝ) : 
  dX = 16 → coverage = 0.5625 → dY = 12 → 
  (π * (dY / 2)^2) = coverage * (π * (dX / 2)^2) := by
  sorry

end frame_diameter_l419_41975


namespace license_plate_count_l419_41920

/-- The number of vowels (excluding Y) -/
def num_vowels : Nat := 5

/-- The number of digits between 1 and 5 -/
def num_digits : Nat := 5

/-- The number of consonants (including Y) -/
def num_consonants : Nat := 26 - num_vowels

/-- The total number of license plates meeting the specified criteria -/
def total_plates : Nat := num_vowels * num_digits * num_consonants * num_consonants * num_vowels

theorem license_plate_count : total_plates = 55125 := by
  sorry

end license_plate_count_l419_41920


namespace polynomial_product_sum_l419_41964

theorem polynomial_product_sum (k j : ℚ) : 
  (∀ d, (8*d^2 - 4*d + k) * (4*d^2 + j*d - 10) = 32*d^4 - 56*d^3 - 68*d^2 + 28*d - 90) →
  k + j = 23/3 := by
sorry

end polynomial_product_sum_l419_41964


namespace field_day_shirt_cost_l419_41926

/-- The total cost of shirts for field day -/
def total_cost (kindergarten_count : ℕ) (kindergarten_price : ℚ)
                (first_grade_count : ℕ) (first_grade_price : ℚ)
                (second_grade_count : ℕ) (second_grade_price : ℚ)
                (third_grade_count : ℕ) (third_grade_price : ℚ) : ℚ :=
  kindergarten_count * kindergarten_price +
  first_grade_count * first_grade_price +
  second_grade_count * second_grade_price +
  third_grade_count * third_grade_price

/-- The total cost of shirts for field day is $2317.00 -/
theorem field_day_shirt_cost :
  total_cost 101 (580/100) 113 5 107 (560/100) 108 (525/100) = 2317 := by
  sorry

end field_day_shirt_cost_l419_41926


namespace triple_composition_even_l419_41991

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem triple_composition_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
by sorry

end triple_composition_even_l419_41991


namespace rect_to_cylindrical_conversion_l419_41970

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical_conversion (x y z : ℝ) :
  x = 3 ∧ y = -3 * Real.sqrt 3 ∧ z = 5 →
  ∃ (r θ : ℝ),
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r = 6 ∧
    θ = 4 * Real.pi / 3 ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ ∧
    z = 5 :=
by sorry

end rect_to_cylindrical_conversion_l419_41970


namespace sqrt_2x_minus_4_meaningful_l419_41965

theorem sqrt_2x_minus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 4) ↔ x ≥ 2 := by sorry

end sqrt_2x_minus_4_meaningful_l419_41965


namespace inscribed_square_side_length_l419_41989

/-- An isosceles right triangle with an inscribed square -/
structure IsoscelesRightTriangleWithSquare where
  /-- Length of the leg of the isosceles right triangle -/
  a : ℝ
  /-- Side length of the inscribed square -/
  s : ℝ
  /-- The triangle is isosceles and right-angled -/
  isIsoscelesRight : True
  /-- The square is inscribed with one vertex on the hypotenuse -/
  squareOnHypotenuse : True
  /-- The square has one vertex at the right angle of the triangle -/
  squareAtRightAngle : True
  /-- The square has two vertices on the legs of the triangle -/
  squareOnLegs : True
  /-- The leg length is positive -/
  a_pos : 0 < a

/-- The side length of the inscribed square is half the leg length of the triangle -/
theorem inscribed_square_side_length 
  (triangle : IsoscelesRightTriangleWithSquare) : 
  triangle.s = triangle.a / 2 := by
  sorry


end inscribed_square_side_length_l419_41989


namespace sqrt_expression_equals_three_l419_41923

theorem sqrt_expression_equals_three : 
  (Real.sqrt 3 - 2) * Real.sqrt 3 + Real.sqrt 12 = 3 := by
  sorry

end sqrt_expression_equals_three_l419_41923


namespace distance_ratio_l419_41992

def travel_scenario (x y w : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ w > 0 ∧ y / w = x / w + (x + y) / (5 * w)

theorem distance_ratio (x y w : ℝ) (h : travel_scenario x y w) : x / y = 2 / 3 := by
  sorry

end distance_ratio_l419_41992


namespace set_A_equals_singleton_l419_41955

-- Define the set A
def A : Set (ℕ × ℕ) := {p | p.1 > 0 ∧ p.2 > 0 ∧ p.2 = 6 / (p.1 + 3)}

-- State the theorem
theorem set_A_equals_singleton : A = {(3, 1)} := by sorry

end set_A_equals_singleton_l419_41955


namespace hyperbola_focal_distance_l419_41972

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  /-- The absolute difference of distances from a point on the hyperbola to the foci -/
  vertex_distance : ℝ
  /-- The eccentricity of the hyperbola -/
  eccentricity : ℝ

/-- Calculates the length of the focal distance of a hyperbola -/
def focal_distance (h : Hyperbola) : ℝ :=
  h.vertex_distance * h.eccentricity

/-- Theorem stating that for a hyperbola with given properties, the focal distance is 10 -/
theorem hyperbola_focal_distance :
  ∀ h : Hyperbola, h.vertex_distance = 6 ∧ h.eccentricity = 5/3 → focal_distance h = 10 := by
  sorry

end hyperbola_focal_distance_l419_41972


namespace factorization_problem_1_factorization_problem_2_l419_41921

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  3 * x^2 * y + 12 * x^2 * y^2 + 12 * x * y^3 = 3 * x * y * (x + 4 * x * y + 4 * y^2) := by
  sorry

-- Problem 2
theorem factorization_problem_2 (a b : ℝ) :
  2 * a^5 * b - 2 * a * b^5 = 2 * a * b * (a^2 + b^2) * (a + b) * (a - b) := by
  sorry

end factorization_problem_1_factorization_problem_2_l419_41921


namespace algebraic_simplification_l419_41954

theorem algebraic_simplification (x y : ℝ) (h : y ≠ 0) :
  (25 * x^3 * y) * (8 * x * y) * (1 / (5 * x * y^2)^2) = 8 * x^2 / y^2 := by
  sorry

end algebraic_simplification_l419_41954


namespace f_equals_g_l419_41900

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := (x^6)^(1/3)

-- Theorem statement
theorem f_equals_g : f = g := by sorry

end f_equals_g_l419_41900


namespace reciprocal_difference_problem_l419_41917

theorem reciprocal_difference_problem (m : ℚ) (hm : m ≠ 1) (h : 1 / (m - 1) = m) :
  m^4 + 1 / m^4 = 7 := by
  sorry

end reciprocal_difference_problem_l419_41917


namespace journey_time_ratio_l419_41946

/-- Proves that the ratio of the time taken for the journey back to the time taken for the journey to San Francisco is 3:2, given the average speeds -/
theorem journey_time_ratio (distance : ℝ) (speed_to_sf : ℝ) (avg_speed : ℝ)
  (h1 : speed_to_sf = 45)
  (h2 : avg_speed = 30)
  (h3 : distance > 0) :
  (distance / avg_speed - distance / speed_to_sf) / (distance / speed_to_sf) = 1/2 :=
by sorry

end journey_time_ratio_l419_41946


namespace perpendicular_lines_from_parallel_planes_l419_41939

/-- A structure representing a 3D geometric space with lines and planes. -/
structure GeometricSpace where
  Line : Type
  Plane : Type
  parallelLinePlane : Line → Plane → Prop
  perpendicularLinePlane : Line → Plane → Prop
  parallelPlanes : Plane → Plane → Prop
  perpendicularLines : Line → Line → Prop

/-- Theorem stating the relationship between parallel planes and perpendicular lines. -/
theorem perpendicular_lines_from_parallel_planes 
  (S : GeometricSpace) 
  (α β : S.Plane) 
  (m n : S.Line) :
  S.parallelPlanes α β →
  S.perpendicularLinePlane m α →
  S.parallelLinePlane n β →
  S.perpendicularLines m n :=
sorry

end perpendicular_lines_from_parallel_planes_l419_41939


namespace cooper_fence_bricks_l419_41983

/-- Represents the dimensions of a wall in bricks -/
structure WallDimensions where
  length : Nat
  height : Nat
  depth : Nat

/-- Calculates the number of bricks needed for a wall -/
def bricksForWall (wall : WallDimensions) : Nat :=
  wall.length * wall.height * wall.depth

/-- The dimensions of Cooper's four walls -/
def wall1 : WallDimensions := { length := 15, height := 6, depth := 3 }
def wall2 : WallDimensions := { length := 20, height := 4, depth := 2 }
def wall3 : WallDimensions := { length := 25, height := 5, depth := 3 }
def wall4 : WallDimensions := { length := 17, height := 7, depth := 2 }

/-- Theorem: The total number of bricks needed for Cooper's fence is 1043 -/
theorem cooper_fence_bricks :
  bricksForWall wall1 + bricksForWall wall2 + bricksForWall wall3 + bricksForWall wall4 = 1043 := by
  sorry

end cooper_fence_bricks_l419_41983


namespace median_and_area_of_triangle_l419_41938

/-- Triangle DEF with given side lengths -/
structure Triangle where
  DE : ℝ
  DF : ℝ
  EF : ℝ

/-- The isosceles triangle DEF with given side lengths -/
def isoscelesTriangle : Triangle where
  DE := 13
  DF := 13
  EF := 14

/-- The length of the median DM in triangle DEF -/
def medianLength (t : Triangle) : ℝ := sorry

/-- The area of triangle DEF -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Theorem stating the length of the median and the area of the triangle -/
theorem median_and_area_of_triangle :
  medianLength isoscelesTriangle = 2 * Real.sqrt 30 ∧
  triangleArea isoscelesTriangle = 84 := by sorry

end median_and_area_of_triangle_l419_41938


namespace smallest_y_divisible_by_11_l419_41935

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def number_with_y (y : ℕ) : ℕ :=
  7000000 + y * 100000 + 86038

theorem smallest_y_divisible_by_11 :
  ∀ y : ℕ, y < 14 → ¬(is_divisible_by_11 (number_with_y y)) ∧
  is_divisible_by_11 (number_with_y 14) := by
  sorry

end smallest_y_divisible_by_11_l419_41935


namespace mobile_wire_left_l419_41956

/-- The amount of wire left after making mobiles -/
def wire_left (total_wire : ℚ) (wire_per_mobile : ℚ) : ℚ :=
  total_wire - wire_per_mobile * ⌊total_wire / wire_per_mobile⌋

/-- Converts millimeters to centimeters -/
def mm_to_cm (mm : ℚ) : ℚ :=
  mm / 10

theorem mobile_wire_left : 
  mm_to_cm (wire_left 117.6 4) = 0.16 := by
  sorry

end mobile_wire_left_l419_41956


namespace square_sum_inequality_l419_41974

theorem square_sum_inequality (a b : ℝ) : a^2 + b^2 ≥ 2*(a - b - 1) := by
  sorry

end square_sum_inequality_l419_41974


namespace angle_A_measure_l419_41950

/-- In a geometric configuration with angles of 110°, 100°, and 40°, there exists an angle A that measures 30°. -/
theorem angle_A_measure (α β γ : Real) (h1 : α = 110) (h2 : β = 100) (h3 : γ = 40) :
  ∃ A : Real, A = 30 := by
  sorry

end angle_A_measure_l419_41950


namespace polygon_interior_angles_l419_41924

theorem polygon_interior_angles (n : ℕ) : 
  (n - 2) * 180 = 540 → n = 5 := by
  sorry

end polygon_interior_angles_l419_41924


namespace classroom_pencils_l419_41901

/-- The number of pencils a teacher gives out to a classroom of students. -/
def pencils_given_out (num_students : ℕ) (dozens_per_student : ℕ) (pencils_per_dozen : ℕ) : ℕ :=
  num_students * dozens_per_student * pencils_per_dozen

/-- Theorem stating the total number of pencils given out in the classroom scenario. -/
theorem classroom_pencils : 
  pencils_given_out 96 7 12 = 8064 := by
  sorry

end classroom_pencils_l419_41901


namespace third_number_solution_l419_41909

theorem third_number_solution (x : ℝ) : 3 + 33 + x + 33.3 = 399.6 → x = 330.3 := by
  sorry

end third_number_solution_l419_41909


namespace number_of_boys_l419_41981

/-- The number of boys in a school with the given conditions -/
theorem number_of_boys (total : ℕ) (boys : ℕ) : 
  total = 400 → 
  boys + (boys * total) / 100 = total →
  boys = 80 :=
by sorry

end number_of_boys_l419_41981


namespace trigonometric_identity_l419_41908

theorem trigonometric_identity (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : Real.cos (α + Real.pi / 6) = 4 / 5) : 
  Real.sin (2 * α + Real.pi / 3) = 24 / 25 := by
  sorry

end trigonometric_identity_l419_41908


namespace gcd_10293_29384_l419_41985

theorem gcd_10293_29384 : Nat.gcd 10293 29384 = 1 := by
  sorry

end gcd_10293_29384_l419_41985


namespace total_spent_on_cards_l419_41933

def digimon_pack_price : ℚ := 4.45
def digimon_pack_count : ℕ := 4
def baseball_deck_price : ℚ := 6.06

theorem total_spent_on_cards :
  digimon_pack_price * digimon_pack_count + baseball_deck_price = 23.86 := by
  sorry

end total_spent_on_cards_l419_41933


namespace calculate_expression_l419_41928

theorem calculate_expression : ((9^9 / 9^8)^2 * 3^4) / 2^4 = 410 + 1/16 := by
  sorry

end calculate_expression_l419_41928


namespace smallest_perfect_square_divisible_by_4_and_5_l419_41932

theorem smallest_perfect_square_divisible_by_4_and_5 :
  ∀ n : ℕ, n > 0 → n.sqrt ^ 2 = n → n % 4 = 0 → n % 5 = 0 → n ≥ 400 :=
by sorry

end smallest_perfect_square_divisible_by_4_and_5_l419_41932


namespace min_value_h_l419_41915

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

noncomputable def g (x : ℝ) : ℝ := x * Real.exp x

noncomputable def h (x : ℝ) : ℝ := f x / g x

theorem min_value_h :
  ∃ (min : ℝ), min = 2 / Real.pi ∧
  ∀ x ∈ Set.Ioo 0 (Real.pi / 2), h x ≥ min :=
by sorry

end min_value_h_l419_41915


namespace geometric_series_ratio_l419_41986

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4) / (1 - r)) → r = 1/2 := by
  sorry

end geometric_series_ratio_l419_41986


namespace maximum_marks_calculation_l419_41942

theorem maximum_marks_calculation (percentage : ℝ) (obtained_marks : ℝ) (max_marks : ℝ) : 
  percentage = 95 → obtained_marks = 285 → 
  (obtained_marks / max_marks) * 100 = percentage → 
  max_marks = 300 := by
  sorry

end maximum_marks_calculation_l419_41942


namespace min_draw_count_correct_l419_41912

/-- Represents a box of colored balls -/
structure ColoredBallBox where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat

/-- The setup of the two boxes -/
def box1 : ColoredBallBox := ⟨40, 30, 25, 15⟩
def box2 : ColoredBallBox := ⟨35, 25, 20, 0⟩

/-- The target number of balls of a single color -/
def targetCount : Nat := 20

/-- The minimum number of balls to draw -/
def minDrawCount : Nat := 73

/-- Theorem stating the minimum number of balls to draw -/
theorem min_draw_count_correct : 
  ∀ (draw : Nat), draw < minDrawCount → 
  ∃ (redCount greenCount yellowCount blueCount : Nat),
    redCount < targetCount ∧
    greenCount < targetCount ∧
    yellowCount < targetCount ∧
    blueCount < targetCount ∧
    redCount + greenCount + yellowCount + blueCount = draw ∧
    redCount ≤ box1.red + box2.red ∧
    greenCount ≤ box1.green + box2.green ∧
    yellowCount ≤ box1.yellow + box2.yellow ∧
    blueCount ≤ box1.blue + box2.blue :=
by sorry

#check min_draw_count_correct

end min_draw_count_correct_l419_41912


namespace thirteenth_result_l419_41903

theorem thirteenth_result (total_count : Nat) (total_avg : ℚ) (first_12_avg : ℚ) (last_12_avg : ℚ) 
  (h_total_count : total_count = 25)
  (h_total_avg : total_avg = 20)
  (h_first_12_avg : first_12_avg = 14)
  (h_last_12_avg : last_12_avg = 17) :
  ∃ (thirteenth : ℚ), 
    (total_count : ℚ) * total_avg = 
      12 * first_12_avg + thirteenth + 12 * last_12_avg ∧ 
    thirteenth = 128 := by
  sorry

end thirteenth_result_l419_41903


namespace red_peaches_count_l419_41994

/-- Represents a basket of peaches -/
structure Basket :=
  (total : ℕ)
  (green : ℕ)
  (h_green_le_total : green ≤ total)

/-- Calculates the number of red peaches in a basket -/
def red_peaches (b : Basket) : ℕ := b.total - b.green

/-- Theorem: The number of red peaches in a basket with 10 total peaches and 3 green peaches is 7 -/
theorem red_peaches_count (b : Basket) (h_total : b.total = 10) (h_green : b.green = 3) : 
  red_peaches b = 7 := by
  sorry

end red_peaches_count_l419_41994


namespace root_product_expression_l419_41953

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) 
  (h1 : α^2 + p*α + 2 = 0) 
  (h2 : β^2 + p*β + 2 = 0)
  (h3 : γ^2 + q*γ + 2 = 0)
  (h4 : δ^2 + q*δ + 2 = 0) :
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = -2*(p^2 - q^2) + 4 := by
  sorry

end root_product_expression_l419_41953


namespace total_turtles_count_l419_41927

/-- The number of turtles Martha received -/
def martha_turtles : ℕ := 40

/-- The number of turtles Marion received -/
def marion_turtles : ℕ := martha_turtles + 20

/-- The total number of turtles received by Marion and Martha -/
def total_turtles : ℕ := marion_turtles + martha_turtles

theorem total_turtles_count : total_turtles = 100 := by
  sorry

end total_turtles_count_l419_41927


namespace bits_of_base16_ABCD_l419_41944

/-- The number of bits in the binary representation of a base-16 number ABCD₁₆ --/
theorem bits_of_base16_ABCD : ∃ (A B C D : ℕ), 
  A < 16 ∧ B < 16 ∧ C < 16 ∧ D < 16 →
  let base16_value := A * 16^3 + B * 16^2 + C * 16^1 + D * 16^0
  let binary_repr := Nat.bits base16_value
  binary_repr.length = 16 := by
  sorry

end bits_of_base16_ABCD_l419_41944


namespace parallel_vectors_k_equals_two_l419_41976

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, prove that if they are parallel, then k = 2 -/
theorem parallel_vectors_k_equals_two (k : ℝ) :
  let a : ℝ × ℝ := (k - 1, k)
  let b : ℝ × ℝ := (1, 2)
  are_parallel a b → k = 2 := by
  sorry

end parallel_vectors_k_equals_two_l419_41976


namespace green_light_most_probable_l419_41971

-- Define the durations of each light
def red_duration : ℕ := 30
def yellow_duration : ℕ := 5
def green_duration : ℕ := 40

-- Define the total cycle duration
def total_duration : ℕ := red_duration + yellow_duration + green_duration

-- Define the probabilities of encountering each light
def prob_red : ℚ := red_duration / total_duration
def prob_yellow : ℚ := yellow_duration / total_duration
def prob_green : ℚ := green_duration / total_duration

-- Theorem: The probability of encountering a green light is higher than the other lights
theorem green_light_most_probable : 
  prob_green > prob_red ∧ prob_green > prob_yellow :=
sorry

end green_light_most_probable_l419_41971


namespace prime_factor_difference_l419_41977

theorem prime_factor_difference (n : Nat) (h : n = 278459) :
  ∃ (p q : Nat), Prime p ∧ Prime q ∧ p ∣ n ∧ q ∣ n ∧
  (∀ r : Nat, Prime r → r ∣ n → p ≥ r ∧ r ≥ q) ∧
  p - q = 254 := by
  sorry

end prime_factor_difference_l419_41977


namespace sufficient_not_necessary_l419_41904

theorem sufficient_not_necessary (p q : Prop) :
  (∃ (h : p ∧ q), ¬p = False) ∧
  (∃ (h : ¬p = False), ¬(p ∧ q = True)) :=
sorry

end sufficient_not_necessary_l419_41904


namespace balance_theorem_l419_41945

/-- Represents the weight of a ball in terms of blue balls -/
@[ext] structure BallWeight where
  blue : ℚ

/-- The weight of a red ball in terms of blue balls -/
def red_weight : BallWeight := ⟨2⟩

/-- The weight of a yellow ball in terms of blue balls -/
def yellow_weight : BallWeight := ⟨3⟩

/-- The weight of a white ball in terms of blue balls -/
def white_weight : BallWeight := ⟨5/3⟩

/-- The weight of a blue ball in terms of blue balls -/
def blue_weight : BallWeight := ⟨1⟩

theorem balance_theorem :
  2 * red_weight.blue + 4 * yellow_weight.blue + 3 * white_weight.blue = 21 * blue_weight.blue :=
by sorry

end balance_theorem_l419_41945


namespace tan_theta_plus_pi_over_8_minus_reciprocal_l419_41910

theorem tan_theta_plus_pi_over_8_minus_reciprocal (θ : Real) 
  (h : 3 * Real.sin θ + Real.cos θ = Real.sqrt 10) : 
  Real.tan (θ + π/8) - (1 / Real.tan (θ + π/8)) = -14 := by
  sorry

end tan_theta_plus_pi_over_8_minus_reciprocal_l419_41910


namespace worker_savings_percentage_l419_41914

theorem worker_savings_percentage
  (last_year_salary : ℝ)
  (last_year_savings_percentage : ℝ)
  (this_year_salary_increase : ℝ)
  (this_year_savings_percentage : ℝ)
  (h1 : this_year_salary_increase = 0.20)
  (h2 : this_year_savings_percentage = 0.05)
  (h3 : this_year_savings_percentage * (1 + this_year_salary_increase) * last_year_salary = last_year_savings_percentage * last_year_salary)
  : last_year_savings_percentage = 0.06 := by
  sorry

end worker_savings_percentage_l419_41914


namespace x_value_l419_41905

theorem x_value : ∃ x : ℝ, x = 88 * (1 + 0.40) ∧ x = 123.2 :=
by sorry

end x_value_l419_41905


namespace pirate_coin_division_l419_41947

theorem pirate_coin_division (n m : ℕ) : 
  n % 10 = 5 → m = 2 * n → m % 10 = 0 := by
  sorry

end pirate_coin_division_l419_41947


namespace sound_travel_distance_l419_41958

/-- The speed of sound in air at 20°C in meters per second -/
def speed_of_sound_at_20C : ℝ := 342

/-- The time of travel in seconds -/
def travel_time : ℝ := 5

/-- The distance traveled by sound in 5 seconds at 20°C -/
def distance_traveled : ℝ := speed_of_sound_at_20C * travel_time

theorem sound_travel_distance : distance_traveled = 1710 := by
  sorry

end sound_travel_distance_l419_41958


namespace triangle_inequality_l419_41987

theorem triangle_inequality (R r p : ℝ) (hR : R > 0) (hr : r > 0) (hp : p > 0) :
  16 * R * r - 5 * r^2 ≤ p^2 ∧ p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2 := by
  sorry

end triangle_inequality_l419_41987


namespace odd_numbers_with_difference_16_are_coprime_l419_41995

theorem odd_numbers_with_difference_16_are_coprime 
  (a b : ℤ) 
  (ha : Odd a) 
  (hb : Odd b) 
  (hdiff : |a - b| = 16) : 
  Int.gcd a b = 1 := by
sorry

end odd_numbers_with_difference_16_are_coprime_l419_41995


namespace solution_set_sqrt3_sin_eq_cos_l419_41980

theorem solution_set_sqrt3_sin_eq_cos :
  {x : ℝ | Real.sqrt 3 * Real.sin x = Real.cos x} =
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6} := by
  sorry

end solution_set_sqrt3_sin_eq_cos_l419_41980


namespace cos_angle_relation_l419_41937

theorem cos_angle_relation (α : ℝ) (h : Real.cos (π / 6 + α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 - α) = -(Real.sqrt 3 / 3) := by
  sorry

end cos_angle_relation_l419_41937


namespace divisor_problem_l419_41982

theorem divisor_problem (n : ℕ) (h1 : n = 1025) (h2 : ¬ (n - 4) % 41 = 0) :
  ∀ d : ℕ, d > 41 → d ∣ n → d ∣ (n - 4) :=
sorry

end divisor_problem_l419_41982


namespace equation_solutions_l419_41913

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 - Real.sqrt 2 ∧ x₂ = 2 + Real.sqrt 2 ∧
    x₁^2 - 4*x₁ = 4 ∧ x₂^2 - 4*x₂ = 4) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 2 ∧
    (x₁ + 2)*(x₁ + 1) = 12 ∧ (x₂ + 2)*(x₂ + 1) = 12) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 5/2 ∧ x₂ = 5 ∧
    0.2*x₁^2 + 5/2 = 3/2*x₁ ∧ 0.2*x₂^2 + 5/2 = 3/2*x₂) :=
by sorry

end equation_solutions_l419_41913


namespace mac_loss_calculation_l419_41966

-- Define exchange rates
def canadian_dime_usd : ℝ := 0.075
def canadian_penny_usd : ℝ := 0.0075
def mexican_centavo_usd : ℝ := 0.0045
def cuban_centavo_usd : ℝ := 0.0036
def euro_cent_usd : ℝ := 0.011
def uk_pence_usd : ℝ := 0.013
def canadian_nickel_usd : ℝ := 0.038
def us_half_dollar_usd : ℝ := 0.5
def brazilian_centavo_usd : ℝ := 0.0019
def australian_cent_usd : ℝ := 0.0072
def indian_paisa_usd : ℝ := 0.0013
def mexican_peso_usd : ℝ := 0.045
def japanese_yen_usd : ℝ := 0.0089

-- Define daily trades
def day1_trade : ℝ := 6 * canadian_dime_usd + 2 * canadian_penny_usd
def day2_trade : ℝ := 10 * mexican_centavo_usd + 5 * cuban_centavo_usd
def day3_trade : ℝ := 4 * 0.1 + 1 * euro_cent_usd
def day4_trade : ℝ := 7 * uk_pence_usd + 5 * canadian_nickel_usd
def day5_trade : ℝ := 3 * us_half_dollar_usd + 2 * brazilian_centavo_usd
def day6_trade : ℝ := 12 * australian_cent_usd + 3 * indian_paisa_usd
def day7_trade : ℝ := 8 * mexican_peso_usd + 6 * japanese_yen_usd

-- Define quarter value
def quarter_value : ℝ := 0.25

-- Theorem statement
theorem mac_loss_calculation :
  (day1_trade - quarter_value) +
  (quarter_value - day2_trade) +
  (day3_trade - quarter_value) +
  (day4_trade - quarter_value) +
  (day5_trade - quarter_value) +
  (quarter_value - day6_trade) +
  (day7_trade - quarter_value) = 2.1619 :=
by sorry

end mac_loss_calculation_l419_41966


namespace square_area_from_adjacent_points_l419_41962

/-- Given two adjacent points (2,1) and (2,7) on a square in a Cartesian coordinate plane,
    the area of the square is 36. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (2, 1)
  let p2 : ℝ × ℝ := (2, 7)
  let square_side := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  square_side^2 = 36 :=
by sorry

end square_area_from_adjacent_points_l419_41962


namespace batsman_innings_l419_41963

theorem batsman_innings (average : ℝ) (highest_score : ℝ) (score_difference : ℝ) (average_excluding : ℝ) 
  (h1 : average = 61)
  (h2 : score_difference = 150)
  (h3 : average_excluding = 58)
  (h4 : highest_score = 202) :
  ∃ (n : ℕ), n = 46 ∧ 
    average * n = highest_score + (highest_score - score_difference) + average_excluding * (n - 2) := by
  sorry

end batsman_innings_l419_41963


namespace x_plus_y_values_l419_41998

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 2) (h3 : x > y) :
  x + y = 5 ∨ x + y = 1 :=
by sorry

end x_plus_y_values_l419_41998


namespace ellipse_sum_l419_41936

/-- Represents an ellipse with center (h, k), semi-major axis a, and semi-minor axis c -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  c : ℝ

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.c^2 = 1

theorem ellipse_sum (e : Ellipse) 
    (center_h : e.h = 3)
    (center_k : e.k = -5)
    (major_axis : e.a = 7)
    (minor_axis : e.c = 4) :
  e.h + e.k + e.a + e.c = 9 := by
  sorry

end ellipse_sum_l419_41936


namespace circle_center_sum_l419_41941

/-- Given a circle defined by the equation x^2 + y^2 + 6x - 4y - 12 = 0,
    if (a, b) is the center of this circle, then a + b = -1. -/
theorem circle_center_sum (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 6*x - 4*y - 12 = 0 ↔ (x - a)^2 + (y - b)^2 = (a^2 + b^2 + 6*a - 4*b - 12)) →
  a + b = -1 := by
  sorry

end circle_center_sum_l419_41941


namespace perimeter_unchanged_after_adding_tiles_l419_41929

/-- A configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Represents the addition of tiles to a configuration -/
def add_tiles (config : TileConfiguration) (new_tiles : ℕ) : TileConfiguration :=
  { tiles := config.tiles + new_tiles, perimeter := config.perimeter }

/-- The theorem stating that adding two tiles can maintain the same perimeter -/
theorem perimeter_unchanged_after_adding_tiles :
  ∃ (initial final : TileConfiguration),
    initial.tiles = 9 ∧
    initial.perimeter = 16 ∧
    final = add_tiles initial 2 ∧
    final.perimeter = 16 :=
  sorry

end perimeter_unchanged_after_adding_tiles_l419_41929


namespace exp_greater_than_log_squared_l419_41969

open Real

theorem exp_greater_than_log_squared (x : ℝ) (h : x > 0) : exp x - exp 2 * log x > 0 := by
  sorry

end exp_greater_than_log_squared_l419_41969


namespace unique_number_property_l419_41925

theorem unique_number_property : ∃! x : ℝ, x / 2 = x - 5 := by
  sorry

end unique_number_property_l419_41925


namespace initial_amount_equation_l419_41948

/-- The initial amount Kanul had, in dollars -/
def initial_amount : ℝ := 7058.82

/-- The amount spent on raw materials, in dollars -/
def raw_materials : ℝ := 3000

/-- The amount spent on machinery, in dollars -/
def machinery : ℝ := 2000

/-- The percentage of the initial amount spent as cash -/
def cash_percentage : ℝ := 0.15

/-- The amount spent on labor costs, in dollars -/
def labor_costs : ℝ := 1000

/-- Theorem stating that the initial amount satisfies the equation -/
theorem initial_amount_equation :
  initial_amount = raw_materials + machinery + cash_percentage * initial_amount + labor_costs := by
  sorry

end initial_amount_equation_l419_41948


namespace ratio_equality_l419_41931

theorem ratio_equality : ∃ x : ℚ, (3/4) / (1/2) = x / (2/6) ∧ x = 1/2 := by
  sorry

end ratio_equality_l419_41931


namespace divisible_by_eight_l419_41979

theorem divisible_by_eight (n : ℕ) : 
  8 ∣ (5^n + 2 * 3^(n-1) + 1) := by
  sorry

end divisible_by_eight_l419_41979


namespace dans_marbles_l419_41961

/-- The number of violet marbles Dan has -/
def violet_marbles : ℕ := 64

/-- The number of red marbles Mary gave to Dan -/
def red_marbles : ℕ := 14

/-- The total number of marbles Dan has now -/
def total_marbles : ℕ := violet_marbles + red_marbles

theorem dans_marbles : total_marbles = 78 := by
  sorry

end dans_marbles_l419_41961


namespace clothes_to_total_ratio_l419_41951

def weekly_allowance_1 : ℕ := 5
def weeks_1 : ℕ := 8
def weekly_allowance_2 : ℕ := 6
def weeks_2 : ℕ := 6
def video_game_cost : ℕ := 35
def money_left : ℕ := 3

def total_saved : ℕ := weekly_allowance_1 * weeks_1 + weekly_allowance_2 * weeks_2
def spent_on_video_game_and_left : ℕ := video_game_cost + money_left
def spent_on_clothes : ℕ := total_saved - spent_on_video_game_and_left

theorem clothes_to_total_ratio :
  (spent_on_clothes : ℚ) / total_saved = 1 / 2 := by sorry

end clothes_to_total_ratio_l419_41951


namespace third_graders_count_l419_41922

theorem third_graders_count (T : ℚ) 
  (h1 : T + 2 * T + T / 2 = 70) : T = 20 := by
  sorry

end third_graders_count_l419_41922


namespace product_reciprocal_sum_l419_41907

theorem product_reciprocal_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 12 → (1 / x) = 3 * (1 / y) → x + y = 8 := by
  sorry

end product_reciprocal_sum_l419_41907


namespace mortgage_food_ratio_is_three_to_one_l419_41967

/-- Esperanza's monthly finances -/
structure MonthlyFinances where
  rent : ℕ
  food_ratio : ℚ
  savings : ℕ
  tax_ratio : ℚ
  gross_salary : ℕ

/-- Calculate the ratio of mortgage bill to food expenses -/
def mortgage_to_food_ratio (finances : MonthlyFinances) : ℚ :=
  let food_expense := finances.food_ratio * finances.rent
  let taxes := finances.tax_ratio * finances.savings
  let total_expenses := finances.rent + food_expense + finances.savings + taxes
  let mortgage := finances.gross_salary - total_expenses
  mortgage / food_expense

/-- Theorem stating the ratio of mortgage bill to food expenses -/
theorem mortgage_food_ratio_is_three_to_one :
  let esperanza_finances : MonthlyFinances := {
    rent := 600,
    food_ratio := 3/5,
    savings := 2000,
    tax_ratio := 2/5,
    gross_salary := 4840
  }
  mortgage_to_food_ratio esperanza_finances = 3 := by
  sorry


end mortgage_food_ratio_is_three_to_one_l419_41967


namespace sandy_initial_fish_count_l419_41959

theorem sandy_initial_fish_count (initial_fish final_fish bought_fish : ℕ) 
  (h1 : final_fish = initial_fish + bought_fish)
  (h2 : final_fish = 32)
  (h3 : bought_fish = 6) : 
  initial_fish = 26 := by
sorry

end sandy_initial_fish_count_l419_41959


namespace marker_cost_l419_41973

theorem marker_cost (total_students : Nat) (buyers : Nat) (markers_per_student : Nat) (total_cost : Nat) :
  total_students = 40 →
  buyers > total_students / 2 →
  buyers ≤ total_students →
  markers_per_student % 2 = 0 →
  markers_per_student > 2 →
  total_cost = 3185 →
  ∃ (cost_per_marker : Nat),
    cost_per_marker > markers_per_student ∧
    buyers * markers_per_student * cost_per_marker = total_cost ∧
    cost_per_marker = 13 :=
by sorry

end marker_cost_l419_41973


namespace fruit_store_theorem_l419_41902

def fruit_problem (total_kg : ℕ) (total_cost : ℕ) 
                  (purchase_price_A purchase_price_B : ℕ)
                  (selling_price_A selling_price_B : ℕ) :=
  ∃ (kg_A kg_B : ℕ),
    -- Total kg constraint
    kg_A + kg_B = total_kg ∧ 
    -- Total cost constraint
    kg_A * purchase_price_A + kg_B * purchase_price_B = total_cost ∧
    -- Specific kg values
    kg_A = 65 ∧ kg_B = 75 ∧
    -- Profit calculation
    (kg_A * (selling_price_A - purchase_price_A) + 
     kg_B * (selling_price_B - purchase_price_B)) = 495

theorem fruit_store_theorem : 
  fruit_problem 140 1000 5 9 8 13 := by
  sorry

end fruit_store_theorem_l419_41902


namespace symmetric_point_wrt_x_axis_l419_41930

/-- Given a point A with coordinates (2, 3), its symmetric point with respect to the x-axis has coordinates (2, -3). -/
theorem symmetric_point_wrt_x_axis :
  let A : ℝ × ℝ := (2, 3)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point A = (2, -3) := by
  sorry

end symmetric_point_wrt_x_axis_l419_41930


namespace inequality_problem_l419_41911

theorem inequality_problem (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  ¬(1 / (a - b) > 1 / a) := by
sorry

end inequality_problem_l419_41911


namespace value_of_c_l419_41999

theorem value_of_c (a c : ℕ) (h1 : a = 105) (h2 : a^5 = 3^3 * 5^2 * 7^2 * 11^2 * 13 * c) : c = 385875 := by
  sorry

end value_of_c_l419_41999


namespace sum_of_critical_slopes_l419_41940

/-- Parabola defined by y = 2x^2 -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- Point Q -/
def Q : ℝ × ℝ := (10, -6)

/-- Line through Q with slope m -/
def line (m : ℝ) (x : ℝ) : ℝ := m * (x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, line m x ≠ parabola x

/-- Theorem stating the sum of critical slopes -/
theorem sum_of_critical_slopes :
  ∃ r s, (∀ m, r < m ∧ m < s ↔ no_intersection m) ∧ r + s = 40 := by sorry

end sum_of_critical_slopes_l419_41940


namespace exists_n_factorial_starts_with_2015_l419_41916

/-- Given a natural number n, returns the first four digits of n! as a natural number -/
def firstFourDigitsOfFactorial (n : ℕ) : ℕ :=
  sorry

/-- Theorem: There exists a positive integer n such that the first four digits of n! are 2015 -/
theorem exists_n_factorial_starts_with_2015 : ∃ n : ℕ+, firstFourDigitsOfFactorial n.val = 2015 := by
  sorry

end exists_n_factorial_starts_with_2015_l419_41916


namespace larger_number_problem_l419_41978

theorem larger_number_problem (x y : ℝ) (h1 : y > x) (h2 : 4 * y = 5 * x) (h3 : y - x = 10) : y = 50 := by
  sorry

end larger_number_problem_l419_41978


namespace function_properties_l419_41943

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
def odd_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc a b → f (-x) = -f x

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → f x > f y

-- State the theorem
theorem function_properties :
  odd_on_interval f (-1) 1 ∧ decreasing_on_interval f (-1) 1 →
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → 
    (f x₁ + f x₂) * (x₁ + x₂) ≤ 0) ∧
  (∀ a, f (1 - a) + f ((1 - a)^2) < 0 → a ∈ Set.Ico 0 1) :=
by sorry

end function_properties_l419_41943


namespace rice_mixture_price_l419_41906

/-- Proves that the price of the second type of rice is 9.60 Rs/kg -/
theorem rice_mixture_price (price1 : ℝ) (weight1 : ℝ) (weight2 : ℝ) (mixture_price : ℝ) 
  (h1 : price1 = 6.60)
  (h2 : weight1 = 49)
  (h3 : weight2 = 56)
  (h4 : mixture_price = 8.20)
  (h5 : weight1 + weight2 = 105) :
  ∃ (price2 : ℝ), price2 = 9.60 ∧ 
  (price1 * weight1 + price2 * weight2) / (weight1 + weight2) = mixture_price :=
by sorry

end rice_mixture_price_l419_41906


namespace range_of_f_l419_41949

-- Define the function f
def f (x : ℝ) : ℝ := |1 - x| - |x - 3|

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y ∈ Set.range f, -2 ≤ y ∧ y ≤ 2 ∧
  ∃ x₁ x₂ : ℝ, f x₁ = -2 ∧ f x₂ = 2 :=
by sorry

end range_of_f_l419_41949


namespace square_sum_reciprocal_l419_41990

theorem square_sum_reciprocal (w : ℝ) (hw : w > 0) (heq : w - 1/w = 5) :
  (w + 1/w)^2 = 29 := by
  sorry

end square_sum_reciprocal_l419_41990


namespace smallest_square_containing_circle_l419_41996

theorem smallest_square_containing_circle (r : ℝ) (h : r = 5) : 
  (2 * r) ^ 2 = 100 := by
  sorry

end smallest_square_containing_circle_l419_41996


namespace intersection_of_A_and_B_l419_41984

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l419_41984


namespace cylinder_height_l419_41993

/-- The height of a right cylinder with radius 3 feet and surface area 36π square feet is 3 feet. -/
theorem cylinder_height (π : ℝ) (h : ℝ) : 
  2 * π * 3^2 + 2 * π * 3 * h = 36 * π → h = 3 := by sorry

end cylinder_height_l419_41993


namespace sally_found_two_balloons_l419_41952

/-- The number of additional orange balloons Sally found -/
def additional_balloons (initial final : ℝ) : ℝ := final - initial

/-- Theorem stating that Sally found 2.0 more orange balloons -/
theorem sally_found_two_balloons (initial final : ℝ) 
  (h1 : initial = 9.0) 
  (h2 : final = 11) : 
  additional_balloons initial final = 2.0 := by
  sorry

end sally_found_two_balloons_l419_41952


namespace f_zero_and_range_l419_41934

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * a * x + 3 * a - 4

-- State the theorem
theorem f_zero_and_range :
  -- f(x) has one zero in (-1, 1)
  ∃ (x : ℝ), -1 < x ∧ x < 1 ∧ f a x = 0 →
  -- The range of a
  (12 * (27 - 4 * Real.sqrt 6) / 211 ≤ a ∧ a ≤ 12 * (27 + 4 * Real.sqrt 6) / 211) ∧
  -- When a = 32/17, the solution is 1/2
  (a = 32/17 → f (32/17) (1/2) = 0) :=
sorry


end f_zero_and_range_l419_41934


namespace import_tax_threshold_l419_41997

/-- Proves that the amount in excess of which import tax was applied is $1000 -/
theorem import_tax_threshold (total_value : ℝ) (tax_rate : ℝ) (tax_paid : ℝ) (threshold : ℝ) : 
  total_value = 2570 →
  tax_rate = 0.07 →
  tax_paid = 109.90 →
  tax_rate * (total_value - threshold) = tax_paid →
  threshold = 1000 := by
sorry

end import_tax_threshold_l419_41997


namespace quadratic_expression_value_l419_41957

theorem quadratic_expression_value (x : ℝ) : 2 * x^2 + 3 * x - 1 = 7 → 4 * x^2 + 6 * x + 9 = 25 := by
  sorry

end quadratic_expression_value_l419_41957
