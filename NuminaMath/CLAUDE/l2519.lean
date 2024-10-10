import Mathlib

namespace bookshelf_problem_l2519_251950

/-- Represents the unit price of bookshelf type A -/
def price_A : ℕ := sorry

/-- Represents the unit price of bookshelf type B -/
def price_B : ℕ := sorry

/-- Represents the maximum number of type B bookshelves that can be purchased -/
def max_B : ℕ := sorry

theorem bookshelf_problem :
  (3 * price_A + 2 * price_B = 1020) ∧
  (price_A + 3 * price_B = 900) ∧
  (∀ m : ℕ, m ≤ 20 → price_A * (20 - m) + price_B * m ≤ 4350) →
  (price_A = 180 ∧ price_B = 240 ∧ max_B = 12) :=
by sorry

end bookshelf_problem_l2519_251950


namespace banana_pancakes_count_l2519_251916

/-- The number of banana pancakes given the total, blueberry, and plain pancake counts. -/
def banana_pancakes (total blueberry plain : ℕ) : ℕ :=
  total - blueberry - plain

/-- Theorem stating that the number of banana pancakes is 24 given the specific counts. -/
theorem banana_pancakes_count :
  banana_pancakes 67 20 23 = 24 := by
  sorry

end banana_pancakes_count_l2519_251916


namespace truth_values_of_p_and_q_l2519_251989

theorem truth_values_of_p_and_q (p q : Prop)
  (h1 : p ∨ q)
  (h2 : ¬(p ∧ q))
  (h3 : ¬p) :
  ¬p ∧ q := by sorry

end truth_values_of_p_and_q_l2519_251989


namespace parabola_sum_is_line_l2519_251991

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola -/
def original : QuadraticFunction :=
  { a := 3, b := 4, c := -5 }

/-- Reflects a quadratic function about the x-axis -/
def reflect (f : QuadraticFunction) : QuadraticFunction :=
  { a := -f.a, b := -f.b, c := -f.c }

/-- Translates a quadratic function horizontally -/
def translate (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := f.b - 2 * f.a * d
  , c := f.a * d^2 - f.b * d + f.c }

/-- Adds two quadratic functions -/
def add (f g : QuadraticFunction) : QuadraticFunction :=
  { a := f.a + g.a
  , b := f.b + g.b
  , c := f.c + g.c }

/-- Theorem stating that the sum of the translated original parabola and its reflected and translated version is a non-horizontal line -/
theorem parabola_sum_is_line :
  let f := translate original 4
  let g := translate (reflect original) (-6)
  let sum := add f g
  sum.a = 0 ∧ sum.b ≠ 0 := by sorry

end parabola_sum_is_line_l2519_251991


namespace special_triangle_secant_sum_range_l2519_251935

-- Define a structure for a triangle with the given condition
structure SpecialTriangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = π
  special_condition : A + C = 2 * B

-- Define the secant function
noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

-- State the theorem
theorem special_triangle_secant_sum_range (t : SpecialTriangle) :
  ∃ (f : Real → Real), 
    (∀ x, f x = sec t.A + sec t.C) ∧ 
    (Set.range f = {y | y < -1 ∨ y ≥ 4}) := by
  sorry


end special_triangle_secant_sum_range_l2519_251935


namespace airplane_faster_than_driving_l2519_251979

/-- Proves that taking an airplane is 90 minutes faster than driving for a job interview --/
theorem airplane_faster_than_driving :
  let driving_time_minutes : ℕ := 3 * 60 + 15
  let drive_to_airport : ℕ := 10
  let wait_to_board : ℕ := 20
  let flight_time : ℕ := driving_time_minutes / 3
  let get_off_plane : ℕ := 10
  let total_airplane_time : ℕ := drive_to_airport + wait_to_board + flight_time + get_off_plane
  driving_time_minutes - total_airplane_time = 90 := by
  sorry

end airplane_faster_than_driving_l2519_251979


namespace A_equals_B_l2519_251960

/-- Set A defined as {a | a = 12m + 8n + 4l, m, n, l ∈ ℤ} -/
def A : Set ℤ := {a | ∃ m n l : ℤ, a = 12*m + 8*n + 4*l}

/-- Set B defined as {b | b = 20p + 16q + 12r, p, q, r ∈ ℤ} -/
def B : Set ℤ := {b | ∃ p q r : ℤ, b = 20*p + 16*q + 12*r}

/-- Theorem stating that A = B -/
theorem A_equals_B : A = B := by
  sorry


end A_equals_B_l2519_251960


namespace sqrt_7_simplest_l2519_251907

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≥ 0 → (∃ n : ℕ, y = n ^ 2 * x) → y = x

theorem sqrt_7_simplest :
  is_simplest_quadratic_radical 7 ∧
  ¬ is_simplest_quadratic_radical 9 ∧
  ¬ is_simplest_quadratic_radical 12 ∧
  ¬ is_simplest_quadratic_radical (2/3) :=
sorry

end sqrt_7_simplest_l2519_251907


namespace f_seven_equals_neg_seventeen_l2519_251921

/-- Given a function f(x) = a*x^7 + b*x^3 + c*x - 5 where a, b, and c are constants,
    if f(-7) = 7, then f(7) = -17 -/
theorem f_seven_equals_neg_seventeen 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^7 + b * x^3 + c * x - 5) 
  (h2 : f (-7) = 7) : 
  f 7 = -17 := by
sorry

end f_seven_equals_neg_seventeen_l2519_251921


namespace nina_weekend_earnings_l2519_251908

-- Define the prices and quantities
def necklace_price : ℚ := 25
def bracelet_price : ℚ := 15
def earring_pair_price : ℚ := 10
def ensemble_price : ℚ := 45

def necklaces_sold : ℕ := 5
def bracelets_sold : ℕ := 10
def earring_pairs_sold : ℕ := 20
def ensembles_sold : ℕ := 2

-- Define the total earnings
def total_earnings : ℚ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_pair_price * earring_pairs_sold +
  ensemble_price * ensembles_sold

-- Theorem to prove
theorem nina_weekend_earnings :
  total_earnings = 565 := by
  sorry

end nina_weekend_earnings_l2519_251908


namespace sum_of_possible_base_3_digits_l2519_251992

/-- The number of digits a positive integer has in a given base -/
def num_digits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log base n + 1

/-- Checks if a number has exactly 4 digits in base 7 -/
def has_four_digits_base_7 (n : ℕ) : Prop :=
  num_digits n 7 = 4

/-- The smallest 4-digit number in base 7 -/
def min_four_digit_base_7 : ℕ := 7^3

/-- The largest 4-digit number in base 7 -/
def max_four_digit_base_7 : ℕ := 7^4 - 1

/-- The theorem to be proved -/
theorem sum_of_possible_base_3_digits : 
  (∀ n : ℕ, has_four_digits_base_7 n → 
    (num_digits n 3 = 6 ∨ num_digits n 3 = 7)) ∧ 
  (∃ n m : ℕ, has_four_digits_base_7 n ∧ has_four_digits_base_7 m ∧ 
    num_digits n 3 = 6 ∧ num_digits m 3 = 7) :=
sorry

end sum_of_possible_base_3_digits_l2519_251992


namespace jill_first_bus_wait_time_l2519_251953

/-- Represents Jill's bus journey times -/
structure BusJourney where
  first_bus_wait : ℕ
  first_bus_ride : ℕ
  second_bus_ride : ℕ

/-- The conditions of Jill's bus journey -/
def journey_conditions (j : BusJourney) : Prop :=
  j.first_bus_ride = 30 ∧
  j.second_bus_ride = 21 ∧
  j.second_bus_ride * 2 = j.first_bus_wait + j.first_bus_ride

theorem jill_first_bus_wait_time (j : BusJourney) 
  (h : journey_conditions j) : j.first_bus_wait = 12 := by
  sorry

end jill_first_bus_wait_time_l2519_251953


namespace square_root_equation_solution_l2519_251914

theorem square_root_equation_solution (x : ℝ) (h1 : x ≠ 0) (h2 : Real.sqrt ((7 * x) / 5) = x) : x = 7 / 5 := by
  sorry

end square_root_equation_solution_l2519_251914


namespace oomyapeck_eyes_eaten_l2519_251981

/-- The number of eyes Oomyapeck eats given the family size, fish per person, eyes per fish, and eyes given away --/
def eyes_eaten (family_size : ℕ) (fish_per_person : ℕ) (eyes_per_fish : ℕ) (eyes_given_away : ℕ) : ℕ :=
  family_size * fish_per_person * eyes_per_fish - eyes_given_away

/-- Theorem stating that under the given conditions, Oomyapeck eats 22 eyes --/
theorem oomyapeck_eyes_eaten :
  eyes_eaten 3 4 2 2 = 22 := by
  sorry

end oomyapeck_eyes_eaten_l2519_251981


namespace unique_c_value_l2519_251997

-- Define the function f(x) = x⋅(2x+1)
def f (x : ℝ) : ℝ := x * (2 * x + 1)

-- Define the open interval (-2, 3/2)
def interval : Set ℝ := {x | -2 < x ∧ x < 3/2}

-- State the theorem
theorem unique_c_value : ∃! c : ℝ, ∀ x : ℝ, x ∈ interval ↔ f x < c :=
  sorry

end unique_c_value_l2519_251997


namespace projection_of_a_onto_b_l2519_251915

def vector_a : ℝ × ℝ := (-1, 1)
def vector_b : ℝ × ℝ := (3, 4)

theorem projection_of_a_onto_b :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) / Real.sqrt (vector_b.1^2 + vector_b.2^2) = 1/5 := by
  sorry

end projection_of_a_onto_b_l2519_251915


namespace cakes_ratio_l2519_251922

/-- Carter's usual weekly baking schedule -/
def usual_cheesecakes : ℕ := 6
def usual_muffins : ℕ := 5
def usual_red_velvet : ℕ := 8

/-- Total number of cakes Carter usually bakes in a week -/
def usual_total : ℕ := usual_cheesecakes + usual_muffins + usual_red_velvet

/-- Additional cakes baked this week -/
def additional_cakes : ℕ := 38

/-- Theorem stating the ratio of cakes baked this week to usual weeks -/
theorem cakes_ratio :
  ∃ (x : ℕ), x * usual_total = usual_total + additional_cakes ∧
  (x * usual_total : ℚ) / usual_total = 3 := by
  sorry

end cakes_ratio_l2519_251922


namespace point_P_on_circle_O_l2519_251988

/-- A circle with center at the origin and radius 5 -/
def circle_O : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 25}

/-- Point P with coordinates (4,3) -/
def point_P : ℝ × ℝ := (4, 3)

/-- Theorem stating that point P lies on circle O -/
theorem point_P_on_circle_O : point_P ∈ circle_O := by
  sorry

end point_P_on_circle_O_l2519_251988


namespace dividend_proof_l2519_251975

theorem dividend_proof : ∃ (a b : ℕ), 
  (11 * 10^5 + a * 10^3 + 7 * 10^2 + 7 * 10 + b) / 12 = 999809 → 
  11 * 10^5 + a * 10^3 + 7 * 10^2 + 7 * 10 + b = 11997708 :=
by
  sorry

end dividend_proof_l2519_251975


namespace no_function_exists_l2519_251939

theorem no_function_exists : ¬∃ (f : ℤ → ℤ), ∀ (x y z : ℤ), f (x * y) + f (x * z) - f x * f (y * z) ≤ -1 := by
  sorry

end no_function_exists_l2519_251939


namespace lcm_gcf_ratio_240_360_l2519_251910

theorem lcm_gcf_ratio_240_360 : 
  (Nat.lcm 240 360) / (Nat.gcd 240 360) = 6 := by
  sorry

end lcm_gcf_ratio_240_360_l2519_251910


namespace diameter_of_figure_F_l2519_251940

/-- A triangle with semicircles constructed outwardly on each side -/
structure TriangleWithSemicircles where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- The figure F composed of the triangle and the three semicircles -/
def FigureF (t : TriangleWithSemicircles) : Set (ℝ × ℝ) :=
  sorry

/-- The diameter of a set in the plane -/
def diameter (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The diameter of figure F is equal to the semi-perimeter of the triangle -/
theorem diameter_of_figure_F (t : TriangleWithSemicircles) :
    diameter (FigureF t) = (t.a + t.b + t.c) / 2 := by
  sorry

end diameter_of_figure_F_l2519_251940


namespace vector_operations_l2519_251903

/-- Given vectors in ℝ², prove that they are not collinear, find the cosine of the angle between them, and calculate the projection of one vector onto another. -/
theorem vector_operations (a b c : ℝ × ℝ) (h1 : a = (-1, 1)) (h2 : b = (4, 3)) (h3 : c = (5, -2)) :
  ¬ (∃ k : ℝ, a = k • b) ∧
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -Real.sqrt 2 / 10 ∧
  ((a.1 * c.1 + a.2 * c.2) / (a.1^2 + a.2^2)) • a = (7/2 * Real.sqrt 2) • (-1, 1) :=
by sorry

end vector_operations_l2519_251903


namespace necessary_but_not_sufficient_l2519_251937

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x^2 < x → |x - 1| < 2) ∧ 
  (∃ x : ℝ, |x - 1| < 2 ∧ x^2 ≥ x) := by
  sorry

end necessary_but_not_sufficient_l2519_251937


namespace train_length_l2519_251987

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 108 → time = 7 → speed * time * (1000 / 3600) = 210 := by
  sorry

end train_length_l2519_251987


namespace divisibility_of_expression_l2519_251955

theorem divisibility_of_expression (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) 
  (ha_gt_7 : a > 7) (hb_gt_7 : b > 7) : 
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) := by
  sorry

end divisibility_of_expression_l2519_251955


namespace regression_and_variance_l2519_251993

-- Define the data points
def x : List Real := [5, 5.5, 6, 6.5, 7]
def y : List Real := [50, 48, 43, 38, 36]

-- Define the probability of "very good" experience
def p : Real := 0.5

-- Define the number of trials
def n : Nat := 5

-- Theorem statement
theorem regression_and_variance :
  let x_mean := (x.sum) / x.length
  let y_mean := (y.sum) / y.length
  let xy_sum := (List.zip x y).map (fun (a, b) => a * b) |>.sum
  let x_squared_sum := x.map (fun a => a ^ 2) |>.sum
  let slope := (xy_sum - x.length * x_mean * y_mean) / (x_squared_sum - x.length * x_mean ^ 2)
  let intercept := y_mean - slope * x_mean
  let variance := n * p * (1 - p)
  slope = -7.6 ∧ intercept = 88.6 ∧ variance = 5/4 := by
  sorry

#check regression_and_variance

end regression_and_variance_l2519_251993


namespace maximum_of_sum_of_roots_l2519_251980

theorem maximum_of_sum_of_roots (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 13) :
  Real.sqrt (x + 27) + Real.sqrt (13 - x) + Real.sqrt x ≤ 11 ∧
  ∃ y, 0 ≤ y ∧ y ≤ 13 ∧ Real.sqrt (y + 27) + Real.sqrt (13 - y) + Real.sqrt y = 11 :=
by sorry

end maximum_of_sum_of_roots_l2519_251980


namespace adam_shopping_cost_l2519_251954

/-- The total cost of Adam's shopping given the number of sandwiches, 
    price per sandwich, and price of water. -/
def total_cost (num_sandwiches : ℕ) (price_per_sandwich : ℕ) (price_of_water : ℕ) : ℕ :=
  num_sandwiches * price_per_sandwich + price_of_water

/-- Theorem stating that Adam's total shopping cost is $11 -/
theorem adam_shopping_cost : 
  total_cost 3 3 2 = 11 := by
  sorry

end adam_shopping_cost_l2519_251954


namespace ribbon_ratio_l2519_251923

theorem ribbon_ratio : 
  ∀ (original reduced : ℕ), 
  original = 55 → reduced = 35 → 
  (original : ℚ) / (reduced : ℚ) = 11 / 7 := by
sorry

end ribbon_ratio_l2519_251923


namespace sector_central_angle_l2519_251990

/-- Given a sector with radius R and circumference 3R, its central angle is 1 radian -/
theorem sector_central_angle (R : ℝ) (h : R > 0) : 
  let circumference := 3 * R
  let arc_length := circumference - 2 * R
  let central_angle := arc_length / R
  central_angle = 1 := by sorry

end sector_central_angle_l2519_251990


namespace monitor_pixel_count_l2519_251995

/-- Calculate the total number of pixels on a monitor given its dimensions and pixel density. -/
theorem monitor_pixel_count (width : ℕ) (height : ℕ) (pixel_density : ℕ) : 
  width = 32 → height = 18 → pixel_density = 150 → 
  width * height * pixel_density * pixel_density = 12960000 := by
  sorry

end monitor_pixel_count_l2519_251995


namespace tetrahedron_edge_length_is_sqrt_2_l2519_251913

/-- Represents a cube with unit side length -/
structure UnitCube where
  center : ℝ × ℝ × ℝ

/-- Represents a tetrahedron circumscribed around four unit cubes -/
structure Tetrahedron where
  cubes : Fin 4 → UnitCube

/-- The edge length of the tetrahedron -/
def tetrahedron_edge_length (t : Tetrahedron) : ℝ := sorry

/-- The configuration of four unit cubes as described in the problem -/
def cube_configuration : Tetrahedron := sorry

theorem tetrahedron_edge_length_is_sqrt_2 :
  tetrahedron_edge_length cube_configuration = Real.sqrt 2 := by sorry

end tetrahedron_edge_length_is_sqrt_2_l2519_251913


namespace cone_in_cylinder_volume_ratio_l2519_251985

noncomputable def cone_volume (base_area : ℝ) (height : ℝ) : ℝ := 
  (1 / 3) * base_area * height

noncomputable def cylinder_volume (base_area : ℝ) (height : ℝ) : ℝ := 
  base_area * height

theorem cone_in_cylinder_volume_ratio 
  (base_area : ℝ) (height : ℝ) (h_pos : base_area > 0 ∧ height > 0) :
  let v_cone := cone_volume base_area height
  let v_cylinder := cylinder_volume base_area height
  (v_cylinder - v_cone) / v_cone = 2 := by
sorry

end cone_in_cylinder_volume_ratio_l2519_251985


namespace expression_value_l2519_251973

/-- Given x, y, and z as defined, prove that the expression equals 20 -/
theorem expression_value (x y z : ℝ) 
  (hx : x = -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5)
  (hy : y = Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5)
  (hz : z = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5) :
  (x^4 / ((x-y)*(x-z))) + (y^4 / ((y-z)*(y-x))) + (z^4 / ((z-x)*(z-y))) = 20 := by
  sorry

end expression_value_l2519_251973


namespace quadratic_roots_property_l2519_251959

theorem quadratic_roots_property (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) → 
  (3 * e^2 + 4 * e - 7 = 0) → 
  (d - 2) * (e - 2) = 13/3 := by
sorry

end quadratic_roots_property_l2519_251959


namespace max_missed_problems_correct_l2519_251994

/-- The number of problems in the test -/
def total_problems : ℕ := 50

/-- The minimum percentage required to pass the test -/
def pass_percentage : ℚ := 85 / 100

/-- The maximum number of problems a student can miss and still pass the test -/
def max_missed_problems : ℕ := 7

theorem max_missed_problems_correct :
  (max_missed_problems ≤ total_problems) ∧
  ((total_problems - max_missed_problems : ℚ) / total_problems ≥ pass_percentage) ∧
  ∀ n : ℕ, n > max_missed_problems →
    ((total_problems - n : ℚ) / total_problems < pass_percentage) :=
by sorry

end max_missed_problems_correct_l2519_251994


namespace square_area_error_l2519_251982

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := 1.08 * s
  let actual_area := s ^ 2
  let calculated_area := measured_side ^ 2
  let error_percentage := (calculated_area - actual_area) / actual_area * 100
  error_percentage = 16.64 := by
sorry

end square_area_error_l2519_251982


namespace womens_bathing_suits_l2519_251965

theorem womens_bathing_suits (total : ℕ) (mens : ℕ) (womens : ℕ) : 
  total = 19766 → mens = 14797 → womens = total - mens → womens = 4969 := by
  sorry

end womens_bathing_suits_l2519_251965


namespace beaver_carrots_l2519_251900

theorem beaver_carrots :
  ∀ (beaver_burrows rabbit_burrows : ℕ),
    beaver_burrows = rabbit_burrows + 5 →
    5 * beaver_burrows = 7 * rabbit_burrows →
    5 * beaver_burrows = 90 :=
by
  sorry

end beaver_carrots_l2519_251900


namespace tims_soda_cans_l2519_251967

theorem tims_soda_cans (x : ℕ) : 
  x - 6 + (x - 6) / 2 = 24 → x = 22 := by
  sorry

end tims_soda_cans_l2519_251967


namespace homework_question_count_l2519_251919

/-- Calculates the number of true/false questions in a homework assignment -/
theorem homework_question_count (total : ℕ) (mc_ratio : ℕ) (fr_diff : ℕ) (h1 : total = 45) (h2 : mc_ratio = 2) (h3 : fr_diff = 7) : 
  ∃ (tf : ℕ) (fr : ℕ) (mc : ℕ), 
    tf + fr + mc = total ∧ 
    mc = mc_ratio * fr ∧ 
    fr = tf + fr_diff ∧ 
    tf = 6 := by
  sorry

end homework_question_count_l2519_251919


namespace common_ratio_of_sequence_l2519_251952

def geometric_sequence (a : ℤ → ℤ) (r : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem common_ratio_of_sequence (a : ℤ → ℤ) :
  a 0 = 25 ∧ a 1 = -50 ∧ a 2 = 100 ∧ a 3 = -200 →
  ∃ r : ℤ, geometric_sequence a r ∧ r = -2 :=
by
  sorry

end common_ratio_of_sequence_l2519_251952


namespace probability_divisible_by_five_l2519_251969

/-- A three-digit positive integer -/
def ThreeDigitInteger (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- An integer ending in 5 -/
def EndsInFive (n : ℕ) : Prop :=
  n % 10 = 5

/-- The probability that a three-digit positive integer ending in 5 is divisible by 5 is 1 -/
theorem probability_divisible_by_five :
  ∀ n : ℕ, ThreeDigitInteger n → EndsInFive n → n % 5 = 0 :=
sorry

end probability_divisible_by_five_l2519_251969


namespace g_of_8_equals_69_l2519_251920

-- Define the function g
def g (n : ℤ) : ℤ := n^2 - 3*n + 29

-- State the theorem
theorem g_of_8_equals_69 : g 8 = 69 := by
  sorry

end g_of_8_equals_69_l2519_251920


namespace x_value_proof_l2519_251941

theorem x_value_proof (x y z : ℤ) 
  (eq1 : x + y = 20) 
  (eq2 : x - y = 10) 
  (eq3 : x + y + z = 30) : x = 15 := by
  sorry

end x_value_proof_l2519_251941


namespace max_value_of_sum_max_value_achieved_l2519_251962

theorem max_value_of_sum (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 2 → x^2 + y^3 + z^4 ≤ 2 := by
  sorry

theorem max_value_achieved (x y z : ℝ) : 
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧ x^2 + y^3 + z^4 = 2 := by
  sorry

end max_value_of_sum_max_value_achieved_l2519_251962


namespace total_baseball_cards_l2519_251933

theorem total_baseball_cards : 
  let number_of_people : ℕ := 6
  let cards_per_person : ℕ := 8
  number_of_people * cards_per_person = 48 :=
by sorry

end total_baseball_cards_l2519_251933


namespace minimum_cost_for_all_entries_l2519_251929

/-- The cost of a single entry in yuan -/
def entry_cost : ℕ := 2

/-- The number of ways to choose 3 consecutive numbers from 01 to 17 -/
def ways_first_segment : ℕ := 15

/-- The number of ways to choose 2 consecutive numbers from 19 to 29 -/
def ways_second_segment : ℕ := 10

/-- The number of ways to choose 1 number from 30 to 36 -/
def ways_third_segment : ℕ := 7

/-- The total number of possible entries -/
def total_entries : ℕ := ways_first_segment * ways_second_segment * ways_third_segment

/-- The theorem stating the minimum amount of money needed -/
theorem minimum_cost_for_all_entries : 
  entry_cost * total_entries = 2100 := by sorry

end minimum_cost_for_all_entries_l2519_251929


namespace bug_walk_tiles_l2519_251963

/-- The number of tiles a bug visits when walking in a straight line from one corner to the opposite corner of a rectangular floor. -/
def tilesVisited (width : ℕ) (length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- The theorem stating that for a 15x35 foot rectangular floor, a bug walking diagonally visits 45 tiles. -/
theorem bug_walk_tiles : tilesVisited 15 35 = 45 := by
  sorry

end bug_walk_tiles_l2519_251963


namespace cookies_per_box_l2519_251932

/-- The number of cookies Basil consumes per day -/
def cookies_per_day : ℚ := 1/2 + 1/2 + 2

/-- The number of days Basil's cookies should last -/
def days : ℕ := 30

/-- The number of boxes needed for the given number of days -/
def boxes : ℕ := 2

/-- Theorem stating the number of cookies in each box -/
theorem cookies_per_box : 
  (cookies_per_day * days) / boxes = 45 := by sorry

end cookies_per_box_l2519_251932


namespace no_positive_integer_solution_l2519_251961

theorem no_positive_integer_solution (m n : ℕ+) : 4 * m * (m + 1) ≠ n * (n + 1) := by
  sorry

end no_positive_integer_solution_l2519_251961


namespace conic_is_ellipse_l2519_251956

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 12

-- Define the two fixed points
def focus1 : ℝ × ℝ := (0, 2)
def focus2 : ℝ × ℝ := (6, -4)

-- Theorem stating that the equation describes an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ) (center : ℝ × ℝ),
    a > 0 ∧ b > 0 ∧ a > b ∧
    ∀ (x y : ℝ),
      conic_equation x y ↔
        (x - center.1)^2 / a^2 + (y - center.2)^2 / b^2 = 1 :=
sorry

end conic_is_ellipse_l2519_251956


namespace inequality_solution_sets_l2519_251958

open Set

theorem inequality_solution_sets 
  (a b c d : ℝ) 
  (h : {x : ℝ | (b / (x + a)) + ((x + d) / (x + c)) < 0} = Ioo (-1) (-1/3) ∪ Ioo (1/2) 1) :
  {x : ℝ | (b * x / (a * x - 1)) + ((d * x - 1) / (c * x - 1)) < 0} = Ioo 1 3 ∪ Ioo (-2) (-1) :=
sorry

end inequality_solution_sets_l2519_251958


namespace power_function_fixed_point_l2519_251998

theorem power_function_fixed_point (α : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^α
  f 1 = 1 := by sorry

end power_function_fixed_point_l2519_251998


namespace unique_solution_for_exponential_equation_l2519_251974

theorem unique_solution_for_exponential_equation :
  ∀ (a b : ℕ+), 1 + 5^(a : ℕ) = 6^(b : ℕ) ↔ a = 1 ∧ b = 1 := by
  sorry

end unique_solution_for_exponential_equation_l2519_251974


namespace no_natural_numbers_satisfying_condition_l2519_251936

theorem no_natural_numbers_satisfying_condition :
  ∀ (x y : ℕ), x + y - 2021 ≥ Nat.gcd x y + Nat.lcm x y :=
by sorry

end no_natural_numbers_satisfying_condition_l2519_251936


namespace division_of_decimals_l2519_251972

theorem division_of_decimals : (0.045 : ℝ) / 0.0075 = 6 := by
  sorry

end division_of_decimals_l2519_251972


namespace min_nSn_l2519_251924

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  h1 : a 5 = 3  -- a_5 = 3
  h2 : S 10 = 40  -- S_10 = 40

/-- The property that the sequence is arithmetic -/
def isArithmetic (seq : ArithmeticSequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, seq.a (n + 1) = seq.a n + d

/-- The sum function definition -/
def sumProperty (seq : ArithmeticSequence) : Prop :=
  ∀ n : ℕ, seq.S n = (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem min_nSn (seq : ArithmeticSequence) 
  (hArith : isArithmetic seq) (hSum : sumProperty seq) : 
  ∃ m : ℝ, m = -32 ∧ ∀ n : ℕ, (n : ℝ) * seq.S n ≥ m :=
sorry

end min_nSn_l2519_251924


namespace two_face_cubes_4x4x4_l2519_251977

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of unit cubes with exactly two faces on the surface of a cuboid -/
def count_two_face_cubes (c : Cuboid) : ℕ :=
  12 * (c.length - 2)

/-- Theorem: A 4x4x4 cuboid has 24 unit cubes with exactly two faces on its surface -/
theorem two_face_cubes_4x4x4 :
  let c : Cuboid := ⟨4, 4, 4⟩
  count_two_face_cubes c = 24 := by
  sorry

#eval count_two_face_cubes ⟨4, 4, 4⟩

end two_face_cubes_4x4x4_l2519_251977


namespace ratio_problem_l2519_251930

/-- Custom operation @ for positive integers -/
def custom_op (k j : ℕ+) : ℕ+ :=
  sorry

theorem ratio_problem (a b : ℕ+) (t : ℚ) : 
  a = 2020 → t = (a : ℚ) / (b : ℚ) → t = 1/2 → b = 4040 := by
  sorry

end ratio_problem_l2519_251930


namespace toy_count_is_134_l2519_251983

/-- The number of toys initially in the box, given the conditions of the problem -/
def initial_toy_count : ℕ := by sorry

theorem toy_count_is_134 :
  -- Define variables for red and white toys
  ∀ (red white : ℕ),
  -- After removing 2 red toys, red is twice white
  (red - 2 = 2 * white) →
  -- After removing 2 red toys, there are 88 red toys
  (red - 2 = 88) →
  -- The initial toy count is the sum of red and white toys
  initial_toy_count = red + white →
  -- Prove that the initial toy count is 134
  initial_toy_count = 134 := by sorry

end toy_count_is_134_l2519_251983


namespace restaurant_meals_count_l2519_251944

theorem restaurant_meals_count (kids_meals : ℕ) (adult_meals : ℕ) : 
  kids_meals = 8 → 
  2 * adult_meals = kids_meals → 
  kids_meals + adult_meals = 12 := by
  sorry

end restaurant_meals_count_l2519_251944


namespace expectation_of_function_l2519_251968

-- Define the random variable ξ
variable (ξ : ℝ → ℝ)

-- Define the expectation operator
noncomputable def E (X : ℝ → ℝ) : ℝ := sorry

-- Define the variance operator
noncomputable def D (X : ℝ → ℝ) : ℝ := E (fun x => (X x - E X)^2)

theorem expectation_of_function (ξ : ℝ → ℝ) 
  (h1 : E ξ = -1) 
  (h2 : D ξ = 3) : 
  E (fun x => 3 * ((ξ x)^2 - 2)) = 6 := 
sorry

end expectation_of_function_l2519_251968


namespace philip_banana_count_l2519_251917

/-- The number of banana groups in Philip's collection -/
def banana_groups : ℕ := 7

/-- The number of bananas in each group -/
def bananas_per_group : ℕ := 29

/-- The total number of bananas in Philip's collection -/
def total_bananas : ℕ := banana_groups * bananas_per_group

/-- Theorem stating that the total number of bananas is 203 -/
theorem philip_banana_count : total_bananas = 203 := by
  sorry

end philip_banana_count_l2519_251917


namespace prime_sum_theorem_l2519_251949

theorem prime_sum_theorem (p q r s : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧  -- p, q, r, s are primes
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧  -- p, q, r, s are distinct
  Prime (p + q + r + s) ∧  -- their sum is prime
  ∃ a, p^2 + q*r = a^2 ∧  -- p² + qr is a perfect square
  ∃ b, p^2 + q*s = b^2  -- p² + qs is a perfect square
  → p + q + r + s = 23 := by
sorry

end prime_sum_theorem_l2519_251949


namespace sue_votes_count_l2519_251957

def total_votes : ℕ := 1000
def candidate1_percentage : ℚ := 20 / 100
def candidate2_percentage : ℚ := 45 / 100

theorem sue_votes_count :
  let sue_percentage : ℚ := 1 - (candidate1_percentage + candidate2_percentage)
  (sue_percentage * total_votes : ℚ) = 350 := by sorry

end sue_votes_count_l2519_251957


namespace no_linear_term_implies_m_equals_six_l2519_251906

theorem no_linear_term_implies_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (2*x + m) * (x - 3) = 2*x^2 - 3*m) → m = 6 := by
sorry

end no_linear_term_implies_m_equals_six_l2519_251906


namespace max_good_permutations_l2519_251984

/-- A sequence of points in the plane is "good" if no three points are collinear,
    the polyline is non-self-intersecting, and each triangle formed by three
    consecutive points is oriented counterclockwise. -/
def is_good_sequence (points : List (ℝ × ℝ)) : Prop :=
  sorry

/-- The number of distinct permutations of n points that form a good sequence -/
def num_good_permutations (n : ℕ) : ℕ :=
  sorry

/-- Theorem: For any integer n ≥ 3, the maximum number of distinct permutations
    of n points in the plane that form a "good" sequence is n^2 - 4n + 6. -/
theorem max_good_permutations (n : ℕ) (h : n ≥ 3) :
  num_good_permutations n = n^2 - 4*n + 6 :=
sorry

end max_good_permutations_l2519_251984


namespace sqrt_equality_condition_l2519_251943

theorem sqrt_equality_condition (a b c : ℕ+) :
  (Real.sqrt (a + b / (c ^ 2 : ℝ)) = a * Real.sqrt (b / (c ^ 2 : ℝ))) ↔ 
  (c ^ 2 : ℝ) = b * (a ^ 2 - 1) / a := by
  sorry

end sqrt_equality_condition_l2519_251943


namespace tangent_line_at_2_g_unique_minimum_l2519_251926

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := x^2 / (1 + x) + 1 / x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f x - 1 / x - a * Real.log x

-- Statement 1: Tangent line equation
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ 23 * x - 36 * y + 20 = 0 :=
sorry

-- Statement 2: Unique minimum point of g
theorem g_unique_minimum (a : ℝ) (h : a > 0) :
  ∃! x, x > 0 ∧ ∀ y, y > 0 → g a y ≥ g a x :=
sorry

end tangent_line_at_2_g_unique_minimum_l2519_251926


namespace pi_approximation_l2519_251901

theorem pi_approximation (π : Real) (h : π = 4 * Real.sin (52 * π / 180)) :
  (1 - 2 * (Real.cos (7 * π / 180))^2) / (π * Real.sqrt (16 - π^2)) = -1/8 := by
  sorry

end pi_approximation_l2519_251901


namespace original_proposition_contrapositive_proposition_both_true_l2519_251925

-- Define the quadratic equation
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + x - m = 0

-- Original proposition
theorem original_proposition : 
  ∀ m : ℝ, m > 0 → has_real_roots m :=
sorry

-- Contrapositive of the original proposition
theorem contrapositive_proposition :
  ∀ m : ℝ, ¬(has_real_roots m) → ¬(m > 0) :=
sorry

-- Both the original proposition and its contrapositive are true
theorem both_true : 
  (∀ m : ℝ, m > 0 → has_real_roots m) ∧ 
  (∀ m : ℝ, ¬(has_real_roots m) → ¬(m > 0)) :=
sorry

end original_proposition_contrapositive_proposition_both_true_l2519_251925


namespace factorization_equality_l2519_251909

theorem factorization_equality (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 3) * (x + 4) - 120 = (x^2 + 5*x + 16) * (x + 6) * (x - 1) := by
  sorry

end factorization_equality_l2519_251909


namespace sum_of_roots_zero_l2519_251971

theorem sum_of_roots_zero (a b c x y : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq1 : a^3 + a*x + y = 0)
  (h_eq2 : b^3 + b*x + y = 0)
  (h_eq3 : c^3 + c*x + y = 0) :
  a + b + c = 0 := by sorry

end sum_of_roots_zero_l2519_251971


namespace factorization_4m_squared_minus_16_l2519_251911

theorem factorization_4m_squared_minus_16 (m : ℝ) :
  4 * m^2 - 16 = 4 * (m + 2) * (m - 2) := by
  sorry

end factorization_4m_squared_minus_16_l2519_251911


namespace abs_negative_two_l2519_251978

theorem abs_negative_two : abs (-2) = 2 := by
  sorry

end abs_negative_two_l2519_251978


namespace safe_lock_configuration_l2519_251902

/-- The number of commission members -/
def n : ℕ := 9

/-- The minimum number of members required to access the safe -/
def k : ℕ := 6

/-- The number of keys for each lock -/
def keys_per_lock : ℕ := n - k + 1

/-- The number of locks needed for the safe -/
def num_locks : ℕ := Nat.choose n (n - k + 1)

theorem safe_lock_configuration :
  num_locks = 126 ∧ keys_per_lock = 4 :=
sorry

end safe_lock_configuration_l2519_251902


namespace bridget_apples_bridget_bought_14_apples_l2519_251927

theorem bridget_apples : ℕ → Prop :=
  fun total : ℕ =>
    let remaining_after_ann : ℕ := total / 2
    let remaining_after_cassie : ℕ := remaining_after_ann - 3
    remaining_after_cassie = 4 → total = 14

-- The proof
theorem bridget_bought_14_apples : bridget_apples 14 := by
  sorry

end bridget_apples_bridget_bought_14_apples_l2519_251927


namespace isosceles_triangle_height_l2519_251966

/-- Given a positive constant s, prove that an isosceles triangle with base 2s
    and area equal to a rectangle with dimensions 2s and s has height 2s. -/
theorem isosceles_triangle_height (s : ℝ) (hs : s > 0) : 
  let rectangle_area := 2 * s * s
  let triangle_base := 2 * s
  let triangle_height := 2 * s
  rectangle_area = 1/2 * triangle_base * triangle_height := by
sorry

end isosceles_triangle_height_l2519_251966


namespace root_sum_fraction_l2519_251948

theorem root_sum_fraction (m₁ m₂ : ℝ) : 
  (∃ a b : ℝ, m₁ * (a^2 - 3*a) + 2*a + 7 = 0 ∧ 
              m₂ * (b^2 - 3*b) + 2*b + 7 = 0 ∧ 
              a/b + b/a = 7/3) →
  m₁/m₂ + m₂/m₁ = 15481/324 := by
sorry

end root_sum_fraction_l2519_251948


namespace cube_expansion_2013_l2519_251945

theorem cube_expansion_2013 : ∃! n : ℕ, 
  n > 0 ∧ 
  (n - 1)^2 + (n - 1) ≤ 2013 ∧ 
  2013 < n^2 + n ∧
  n = 45 := by sorry

end cube_expansion_2013_l2519_251945


namespace triangle_problem_l2519_251964

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  -- Given conditions
  a = 1 ∧ b = 2 ∧ Real.cos C = 1/4 →
  -- Prove
  c = 2 ∧ Real.sin A = Real.sqrt 15 / 8 := by
    sorry

end triangle_problem_l2519_251964


namespace power_function_sum_l2519_251912

/-- A power function passing through (4, 2) has k + a = 3/2 --/
theorem power_function_sum (k a : ℝ) : 
  (∀ x : ℝ, x > 0 → ∃ f : ℝ → ℝ, f x = k * x^a) → 
  k * 4^a = 2 → 
  k + a = 3/2 := by sorry

end power_function_sum_l2519_251912


namespace largest_digit_divisible_by_6_l2519_251986

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def append_digit (n m : ℕ) : ℕ := n * 10 + m

theorem largest_digit_divisible_by_6 :
  ∃ (M : ℕ), M ≤ 9 ∧ 
    is_divisible_by_6 (append_digit 5172 M) ∧ 
    ∀ (K : ℕ), K ≤ 9 → is_divisible_by_6 (append_digit 5172 K) → K ≤ M :=
by
  -- Proof goes here
  sorry

end largest_digit_divisible_by_6_l2519_251986


namespace melanie_football_games_l2519_251942

theorem melanie_football_games (total_games missed_games : ℕ) 
  (h1 : total_games = 7)
  (h2 : missed_games = 4) :
  total_games - missed_games = 3 := by
  sorry

end melanie_football_games_l2519_251942


namespace square_binomial_coefficient_l2519_251999

theorem square_binomial_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 8 * x + 16 = (r * x + s)^2) → a = 1 := by
  sorry

end square_binomial_coefficient_l2519_251999


namespace circle_origin_inside_l2519_251946

theorem circle_origin_inside (m : ℝ) : 
  (∀ x y : ℝ, (x - m)^2 + (y + m)^2 = 8 → (0 : ℝ)^2 + (0 : ℝ)^2 < 8) → 
  -2 < m ∧ m < 2 :=
by sorry

end circle_origin_inside_l2519_251946


namespace problem_statement_l2519_251951

theorem problem_statement (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^2 + 4*b^2 = 1/(a*b) + 3) : 
  (a * b ≤ 1) ∧ 
  (b > a → 1/a^3 - 1/b^3 ≥ 3*(1/a - 1/b)) := by
sorry

end problem_statement_l2519_251951


namespace average_and_difference_l2519_251996

theorem average_and_difference (x : ℝ) : 
  (35 + x) / 2 = 45 → |x - 35| = 20 := by
sorry

end average_and_difference_l2519_251996


namespace other_intersection_point_l2519_251904

/-- Two circles with centers on a line intersecting at two points -/
structure TwoCirclesIntersection where
  -- The line equation: x - y + 1 = 0
  line : ℝ → ℝ → Prop
  line_eq : ∀ x y, line x y ↔ x - y + 1 = 0
  
  -- The circles intersect at two different points
  intersect_points : Fin 2 → ℝ × ℝ
  different_points : intersect_points 0 ≠ intersect_points 1
  
  -- One intersection point is (-2, 2)
  known_point : intersect_points 0 = (-2, 2)

/-- The other intersection point has coordinates (1, -1) -/
theorem other_intersection_point (c : TwoCirclesIntersection) : 
  c.intersect_points 1 = (1, -1) := by
  sorry

end other_intersection_point_l2519_251904


namespace point_not_on_transformed_plane_l2519_251928

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane3D) (k : ℝ) : Plane3D :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def isPointOnPlane (point : Point3D) (plane : Plane3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem to be proved -/
theorem point_not_on_transformed_plane :
  let A : Point3D := { x := -3, y := -2, z := 4 }
  let a : Plane3D := { a := 2, b := -3, c := 1, d := -5 }
  let k : ℝ := -4/5
  let a' : Plane3D := transformPlane a k
  ¬ isPointOnPlane A a' := by
  sorry

end point_not_on_transformed_plane_l2519_251928


namespace perpendicular_equal_diagonals_not_sufficient_for_square_l2519_251905

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of quadrilaterals
def has_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem perpendicular_equal_diagonals_not_sufficient_for_square :
  ∃ (q : Quadrilateral), has_perpendicular_diagonals q ∧ has_equal_diagonals q ∧ ¬is_square q :=
sorry

end perpendicular_equal_diagonals_not_sufficient_for_square_l2519_251905


namespace power_of_power_l2519_251931

theorem power_of_power : (3^4)^2 = 6561 := by sorry

end power_of_power_l2519_251931


namespace race_winner_distance_l2519_251976

theorem race_winner_distance (catrina_distance : ℝ) (catrina_time : ℝ) 
  (sedra_distance : ℝ) (sedra_time : ℝ) (race_distance : ℝ) :
  catrina_distance = 100 ∧ 
  catrina_time = 10 ∧ 
  sedra_distance = 400 ∧ 
  sedra_time = 44 ∧ 
  race_distance = 1000 →
  let catrina_speed := catrina_distance / catrina_time
  let sedra_speed := sedra_distance / sedra_time
  let catrina_race_time := race_distance / catrina_speed
  let sedra_race_distance := sedra_speed * catrina_race_time
  race_distance - sedra_race_distance = 91 :=
by sorry

end race_winner_distance_l2519_251976


namespace hyperbola_eccentricity_l2519_251938

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±√2x,
    prove that its eccentricity is √3 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b / a = Real.sqrt 2) :
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 3 := by sorry

end hyperbola_eccentricity_l2519_251938


namespace ed_doug_marble_difference_l2519_251947

-- Define the initial number of marbles for Ed and Doug
def ed_marbles : ℕ := 45
def doug_initial_marbles : ℕ := ed_marbles - 10

-- Define the number of marbles Doug lost
def doug_lost_marbles : ℕ := 11

-- Define Doug's final number of marbles
def doug_final_marbles : ℕ := doug_initial_marbles - doug_lost_marbles

-- Theorem statement
theorem ed_doug_marble_difference :
  ed_marbles - doug_final_marbles = 21 :=
by sorry

end ed_doug_marble_difference_l2519_251947


namespace completing_square_sum_l2519_251970

-- Define the original quadratic equation
def original_equation (x : ℝ) : Prop := x^2 - 6*x = 1

-- Define the transformed equation
def transformed_equation (x m n : ℝ) : Prop := (x - m)^2 = n

-- Theorem statement
theorem completing_square_sum (m n : ℝ) :
  (∀ x, original_equation x ↔ transformed_equation x m n) →
  m + n = 13 := by
  sorry

end completing_square_sum_l2519_251970


namespace find_x_l2519_251918

-- Define the binary operation ★
def star (a b c d : ℤ) : ℤ × ℤ := (a + c, b - 2*d)

-- Theorem statement
theorem find_x : ∀ x y : ℤ, star (x + 1) (y - 1) 1 3 = (2, -4) → x = 0 := by
  sorry

end find_x_l2519_251918


namespace octagon_area_l2519_251934

/-- The area of an octagon inscribed in a rectangle --/
theorem octagon_area (rectangle_width rectangle_height triangle_base triangle_height : ℝ) 
  (hw : rectangle_width = 5)
  (hh : rectangle_height = 8)
  (htb : triangle_base = 1)
  (hth : triangle_height = 4) :
  rectangle_width * rectangle_height - 4 * (1/2 * triangle_base * triangle_height) = 32 :=
by sorry

end octagon_area_l2519_251934
