import Mathlib

namespace correlation_strength_theorem_l1240_124000

-- Define the correlation coefficient r
def correlation_coefficient (r : ℝ) : Prop := -1 < r ∧ r < 1

-- Define the strength of correlation
def correlation_strength (r : ℝ) : ℝ := |r|

-- Theorem stating the relationship between |r| and correlation strength
theorem correlation_strength_theorem (r : ℝ) (h : correlation_coefficient r) :
  ∀ ε > 0, ∃ δ > 0, ∀ r', correlation_coefficient r' →
    correlation_strength r' < δ → correlation_strength r' < ε :=
sorry

end correlation_strength_theorem_l1240_124000


namespace jacket_price_before_tax_l1240_124088

def initial_amount : ℚ := 13.99
def shirt_price : ℚ := 12.14
def discount_rate : ℚ := 0.05
def additional_money : ℚ := 7.43
def tax_rate : ℚ := 0.10

def discounted_shirt_price : ℚ := shirt_price * (1 - discount_rate)
def money_left : ℚ := initial_amount + additional_money - discounted_shirt_price

theorem jacket_price_before_tax :
  ∃ (x : ℚ), x * (1 + tax_rate) = money_left ∧ x = 8.99 := by sorry

end jacket_price_before_tax_l1240_124088


namespace sum_of_solutions_is_zero_l1240_124079

theorem sum_of_solutions_is_zero :
  let f (x : ℝ) := (-12 * x) / (x^2 - 1) - (3 * x) / (x + 1) + 9 / (x - 1)
  ∃ (a b : ℝ), (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b) ∧ a + b = 0 :=
by sorry

end sum_of_solutions_is_zero_l1240_124079


namespace cubic_equation_root_l1240_124095

theorem cubic_equation_root (c d : ℚ) : 
  (3 + Real.sqrt 5 : ℝ)^3 + c * (3 + Real.sqrt 5)^2 + d * (3 + Real.sqrt 5) + 20 = 0 → 
  d = -26 := by
sorry

end cubic_equation_root_l1240_124095


namespace twenty_five_percent_more_than_eighty_twenty_five_percent_more_than_eighty_proof_l1240_124005

theorem twenty_five_percent_more_than_eighty : ℝ → Prop :=
  fun x => (3/4 * x = 100) → (x = 400/3)

-- The proof is omitted
theorem twenty_five_percent_more_than_eighty_proof : 
  ∃ x : ℝ, twenty_five_percent_more_than_eighty x :=
sorry

end twenty_five_percent_more_than_eighty_twenty_five_percent_more_than_eighty_proof_l1240_124005


namespace chess_tournament_matches_l1240_124012

/-- The number of matches required in a single-elimination tournament -/
def matches_required (num_players : ℕ) : ℕ :=
  num_players - 1

/-- Theorem: A single-elimination tournament with 32 players requires 31 matches -/
theorem chess_tournament_matches :
  matches_required 32 = 31 :=
by
  sorry


end chess_tournament_matches_l1240_124012


namespace spheres_fit_in_box_l1240_124060

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the maximum number of spheres that can fit in a box using a specific packing method -/
noncomputable def maxSpheres (box : BoxDimensions) (sphereDiameter : ℝ) : ℕ :=
  sorry

/-- Theorem stating that 100,000 spheres of 4 cm diameter can fit in the given box -/
theorem spheres_fit_in_box :
  let box : BoxDimensions := ⟨200, 164, 146⟩
  let sphereDiameter : ℝ := 4
  maxSpheres box sphereDiameter ≥ 100000 := by
  sorry

end spheres_fit_in_box_l1240_124060


namespace matching_arrangements_count_l1240_124015

def number_of_people : Nat := 5

/-- The number of arrangements where exactly two people sit in seats matching their numbers -/
def matching_arrangements : Nat :=
  (number_of_people.choose 2) * 2 * 1 * 1

theorem matching_arrangements_count : matching_arrangements = 20 := by
  sorry

end matching_arrangements_count_l1240_124015


namespace least_positive_angle_l1240_124029

theorem least_positive_angle (x a b : ℝ) (h1 : Real.tan x = 2 * a / (3 * b)) 
  (h2 : Real.tan (3 * x) = 3 * b / (2 * a + 3 * b)) :
  x = Real.arctan (2 / 3) ∧ x > 0 ∧ ∀ y, y > 0 → y = Real.arctan (2 / 3) → y ≥ x :=
by sorry

end least_positive_angle_l1240_124029


namespace popping_corn_probability_l1240_124050

theorem popping_corn_probability (total : ℝ) (h_total : total > 0) :
  let white := (3 / 4 : ℝ) * total
  let yellow := (1 / 4 : ℝ) * total
  let white_pop_prob := (3 / 5 : ℝ)
  let yellow_pop_prob := (1 / 2 : ℝ)
  let white_popped := white * white_pop_prob
  let yellow_popped := yellow * yellow_pop_prob
  let total_popped := white_popped + yellow_popped
  (white_popped / total_popped) = (18 / 23 : ℝ) :=
sorry

end popping_corn_probability_l1240_124050


namespace sarahs_age_l1240_124066

/-- Given a person (Sarah) who is 18 years younger than her mother, 
    and the sum of their ages is 50 years, Sarah's age is 16 years. -/
theorem sarahs_age (s m : ℕ) : s = m - 18 ∧ s + m = 50 → s = 16 := by
  sorry

end sarahs_age_l1240_124066


namespace smallest_marble_collection_l1240_124073

theorem smallest_marble_collection (N : ℕ) : 
  N > 1 ∧ 
  N % 9 = 2 ∧ 
  N % 10 = 2 ∧ 
  N % 11 = 2 ∧ 
  (∀ m : ℕ, m > 1 ∧ m % 9 = 2 ∧ m % 10 = 2 ∧ m % 11 = 2 → m ≥ N) →
  N = 992 := by
sorry

end smallest_marble_collection_l1240_124073


namespace corgi_dog_price_calculation_l1240_124037

/-- The price calculation for Corgi dogs with profit --/
theorem corgi_dog_price_calculation (original_price : ℝ) (profit_percentage : ℝ) (num_dogs : ℕ) :
  original_price = 1000 →
  profit_percentage = 30 →
  num_dogs = 2 →
  let profit_per_dog := original_price * (profit_percentage / 100)
  let selling_price_per_dog := original_price + profit_per_dog
  let total_cost := selling_price_per_dog * num_dogs
  total_cost = 2600 := by
  sorry


end corgi_dog_price_calculation_l1240_124037


namespace del_oranges_per_day_l1240_124033

theorem del_oranges_per_day (total : ℕ) (juan : ℕ) (del_days : ℕ) 
  (h_total : total = 107)
  (h_juan : juan = 61)
  (h_del_days : del_days = 2) :
  (total - juan) / del_days = 23 := by
  sorry

end del_oranges_per_day_l1240_124033


namespace concyclic_AQTP_l1240_124099

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the properties and relations
variable (on_circle : Point → Circle → Prop)
variable (intersect : Circle → Circle → Point → Point → Prop)
variable (collinear : Point → Point → Point → Prop)
variable (tangent_intersection : Circle → Point → Point → Point → Prop)
variable (concyclic : Point → Point → Point → Point → Prop)

-- State the theorem
theorem concyclic_AQTP 
  (Γ₁ Γ₂ : Circle) 
  (A B P Q T : Point) :
  intersect Γ₁ Γ₂ A B →
  on_circle P Γ₁ →
  on_circle Q Γ₂ →
  collinear P B Q →
  tangent_intersection Γ₂ P Q T →
  concyclic A Q T P :=
sorry

end concyclic_AQTP_l1240_124099


namespace imaginary_unit_power_2016_l1240_124075

theorem imaginary_unit_power_2016 (i : ℂ) (h : i^2 = -1) : i^2016 = 1 := by
  sorry

end imaginary_unit_power_2016_l1240_124075


namespace count_negative_numbers_l1240_124002

theorem count_negative_numbers : let numbers := [-3^2, (-1)^2006, 0, |(-2)|, -(-2), -3 * 2^2]
  (numbers.filter (· < 0)).length = 2 := by sorry

end count_negative_numbers_l1240_124002


namespace product_coefficients_sum_l1240_124019

theorem product_coefficients_sum (m n : ℚ) : 
  (∀ k : ℚ, (5*k^2 - 4*k + m) * (2*k^2 + n*k - 5) = 10*k^4 - 28*k^3 + 23*k^2 - 18*k + 15) →
  m + n = 35/3 := by
sorry

end product_coefficients_sum_l1240_124019


namespace exists_unique_max_N_l1240_124003

/-- The number of positive divisors of a positive integer -/
def d (n : ℕ+) : ℕ+ := sorry

/-- The function f(n) = d(n) / (n^(1/3)) -/
noncomputable def f (n : ℕ+) : ℝ := (d n : ℝ) / n.val ^ (1/3 : ℝ)

/-- The sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- The theorem stating the existence of a unique N maximizing f(n) -/
theorem exists_unique_max_N : ∃! N : ℕ+, (∀ n : ℕ+, n ≠ N → f N > f n) ∧ sum_of_digits N = 6 := by sorry

end exists_unique_max_N_l1240_124003


namespace swim_time_proof_l1240_124007

-- Define the given constants
def downstream_distance : ℝ := 16
def upstream_distance : ℝ := 10
def still_water_speed : ℝ := 6.5

-- Define the theorem
theorem swim_time_proof :
  ∃ (t c : ℝ),
    t > 0 ∧
    c ≥ 0 ∧
    c < still_water_speed ∧
    downstream_distance / (still_water_speed + c) = t ∧
    upstream_distance / (still_water_speed - c) = t ∧
    t = 2 := by
  sorry

end swim_time_proof_l1240_124007


namespace cuboid_breadth_l1240_124091

/-- The surface area of a cuboid given its length, width, and height -/
def cuboidSurfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The breadth of a cuboid with given length, height, and surface area -/
theorem cuboid_breadth (l h area : ℝ) (hl : l = 8) (hh : h = 12) (harea : area = 960) :
  ∃ w : ℝ, cuboidSurfaceArea l w h = area ∧ w = 19.2 := by sorry

end cuboid_breadth_l1240_124091


namespace banana_distribution_l1240_124084

/-- Given three people with a total of 200 bananas, where one person has 40 more than another
    and the third person has 40 bananas, prove that the person with the least bananas has 60. -/
theorem banana_distribution (total : ℕ) (difference : ℕ) (donna_bananas : ℕ)
    (h_total : total = 200)
    (h_difference : difference = 40)
    (h_donna : donna_bananas = 40) :
    ∃ (lydia dawn : ℕ),
      lydia + dawn + donna_bananas = total ∧
      dawn = lydia + difference ∧
      lydia = 60 := by
  sorry

end banana_distribution_l1240_124084


namespace slower_train_speed_l1240_124016

/-- Prove that the speed of the slower train is 36 km/hr -/
theorem slower_train_speed 
  (train_length : ℝ) 
  (faster_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 55) 
  (h2 : faster_speed = 47) 
  (h3 : passing_time = 36) : 
  ∃ (slower_speed : ℝ), 
    slower_speed = 36 ∧ 
    (2 * train_length) = (faster_speed - slower_speed) * (5/18) * passing_time :=
sorry

end slower_train_speed_l1240_124016


namespace rectangular_prism_volume_range_l1240_124043

theorem rectangular_prism_volume_range (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 9 → 
  a * b + b * c + a * c = 24 → 
  16 ≤ a * b * c ∧ a * b * c ≤ 20 := by
  sorry

end rectangular_prism_volume_range_l1240_124043


namespace revenue_maximized_at_20_l1240_124089

-- Define the revenue function
def R (p : ℝ) : ℝ := p * (160 - 4 * p)

-- State the theorem
theorem revenue_maximized_at_20 :
  ∃ (p_max : ℝ), p_max ≤ 40 ∧ 
  ∀ (p : ℝ), p ≤ 40 → R p ≤ R p_max ∧
  p_max = 20 := by
  sorry

end revenue_maximized_at_20_l1240_124089


namespace conic_common_chords_l1240_124040

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Conic where
  equation : ℝ → ℝ → ℝ

-- Define the problem setup
def are_tangent (c1 c2 : Conic) (p1 p2 : Point) : Prop := sorry

def have_common_points (c1 c2 : Conic) (n : ℕ) : Prop := sorry

def line_through_points (p1 p2 : Point) : Line := sorry

def intersection_point (l1 l2 : Line) : Point := sorry

def common_chord (c1 c2 : Conic) : Line := sorry

def passes_through (l : Line) (p : Point) : Prop := sorry

-- State the theorem
theorem conic_common_chords 
  (Γ Γ₁ Γ₂ : Conic) 
  (A B C D : Point) :
  are_tangent Γ Γ₁ A B →
  are_tangent Γ Γ₂ C D →
  have_common_points Γ₁ Γ₂ 4 →
  ∃ (chord1 chord2 : Line),
    chord1 = common_chord Γ₁ Γ₂ ∧
    chord2 = common_chord Γ₁ Γ₂ ∧
    chord1 ≠ chord2 ∧
    passes_through chord1 (intersection_point (line_through_points A B) (line_through_points C D)) ∧
    passes_through chord2 (intersection_point (line_through_points A B) (line_through_points C D)) :=
by sorry

end conic_common_chords_l1240_124040


namespace pizza_stand_total_slices_l1240_124030

/-- Given the conditions of the pizza stand problem, prove that the total number of slices sold is 5000. -/
theorem pizza_stand_total_slices : 
  let small_price : ℕ := 150
  let large_price : ℕ := 250
  let total_revenue : ℕ := 1050000
  let small_slices_sold : ℕ := 2000
  let large_slices_sold : ℕ := (total_revenue - small_price * small_slices_sold) / large_price
  small_slices_sold + large_slices_sold = 5000 := by
sorry


end pizza_stand_total_slices_l1240_124030


namespace find_y_value_l1240_124097

theorem find_y_value (x y : ℝ) (h1 : 3 * (x^2 + x + 1) = y - 6) (h2 : x = -3) : y = 27 := by
  sorry

end find_y_value_l1240_124097


namespace set_equality_l1240_124013

def U : Finset ℕ := {1,2,3,4,5,6,7,8}
def A : Finset ℕ := {3,4,5}
def B : Finset ℕ := {1,3,6}
def C : Finset ℕ := {2,7,8}

theorem set_equality : C = (C ∪ A) ∩ (C ∪ B) := by
  sorry

end set_equality_l1240_124013


namespace intersection_point_theorem_l1240_124014

/-- Given points A and B, and a point C on the line y=x that intersects AB,
    prove that if AC = 2CB, then the y-coordinate of B is 4. -/
theorem intersection_point_theorem (a : ℝ) : 
  let A : ℝ × ℝ := (7, 1)
  let B : ℝ × ℝ := (1, a)
  let C : ℝ × ℝ := (x, x)
  ∃ x : ℝ, (C.1 - A.1, C.2 - A.2) = 2 • (B.1 - C.1, B.2 - C.2) → a = 4 := by
  sorry

end intersection_point_theorem_l1240_124014


namespace inequality_system_solution_l1240_124025

theorem inequality_system_solution :
  ∀ p : ℝ, (19 * p < 10 ∧ p > 1/2) ↔ (1/2 < p ∧ p < 10/19) := by
  sorry

end inequality_system_solution_l1240_124025


namespace arithmetic_sequence_sum_negative_48_to_0_l1240_124071

def arithmeticSequenceSum (a l d : ℤ) : ℤ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum_negative_48_to_0 :
  arithmeticSequenceSum (-48) 0 2 = -600 := by
  sorry

end arithmetic_sequence_sum_negative_48_to_0_l1240_124071


namespace system_solutions_l1240_124090

-- Define the system of equations
def equation1 (x y : ℝ) : Prop :=
  (-x^7 / y)^(Real.log (-y)) = x^(2 * Real.log (x * y^2))

def equation2 (x y : ℝ) : Prop :=
  y^2 + 2*x*y - 3*x^2 + 12*x + 4*y = 0

-- Define the solution set
def solutions : Set (ℝ × ℝ) :=
  {(2, -2), (3, -9), ((Real.sqrt 17 - 1) / 2, (Real.sqrt 17 - 9) / 2)}

-- Theorem statement
theorem system_solutions :
  ∀ x y : ℝ, x ≠ 0 ∧ y < 0 →
    (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions :=
by sorry

end system_solutions_l1240_124090


namespace complex_equation_solution_l1240_124034

theorem complex_equation_solution :
  ∀ z : ℂ, z - 3 * I = 3 + z * I → z = -3 := by
  sorry

end complex_equation_solution_l1240_124034


namespace needle_cylinder_height_gt_six_l1240_124093

/-- Represents the properties of a cylinder formed by needles piercing a skein of yarn -/
structure NeedleCylinder where
  num_needles : ℕ
  needle_radius : ℝ
  cylinder_radius : ℝ

/-- The theorem stating that the height of the cylinder must be greater than 6 -/
theorem needle_cylinder_height_gt_six (nc : NeedleCylinder)
  (h_num_needles : nc.num_needles = 72)
  (h_needle_radius : nc.needle_radius = 1)
  (h_cylinder_radius : nc.cylinder_radius = 6) :
  ∀ h : ℝ, h > 6 → 
    2 * π * nc.cylinder_radius^2 + 2 * π * nc.cylinder_radius * h > 
    2 * π * nc.num_needles * nc.needle_radius^2 + 2 * π * nc.cylinder_radius^2 :=
by sorry

end needle_cylinder_height_gt_six_l1240_124093


namespace semicircle_perimeter_l1240_124076

/-- The perimeter of a semicircle with radius 4.8 cm is equal to π * 4.8 + 9.6 cm. -/
theorem semicircle_perimeter (π : ℝ) (h : π = Real.pi) :
  let r : ℝ := 4.8
  (π * r + 2 * r) = π * 4.8 + 9.6 :=
by sorry

end semicircle_perimeter_l1240_124076


namespace sqrt_x_div_sqrt_y_l1240_124069

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  (((1 : ℝ) / 3) ^ 2 + ((1 : ℝ) / 4) ^ 2) / (((1 : ℝ) / 5) ^ 2 + ((1 : ℝ) / 6) ^ 2) = 25 * x / (53 * y) →
  Real.sqrt x / Real.sqrt y = 150 / 239 := by
  sorry

end sqrt_x_div_sqrt_y_l1240_124069


namespace even_odd_sum_zero_l1240_124065

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function g is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

/-- Main theorem: If f is even and g(x) = f(x-1) is odd, then f(2009) + f(2011) = 0 -/
theorem even_odd_sum_zero (f : ℝ → ℝ) (g : ℝ → ℝ) 
    (h_even : IsEven f) (h_odd : IsOdd g) (h_g : ∀ x, g x = f (x - 1)) :
    f 2009 + f 2011 = 0 := by
  sorry

end even_odd_sum_zero_l1240_124065


namespace max_sum_with_constraints_l1240_124067

theorem max_sum_with_constraints (a b : ℝ) 
  (h1 : 5 * a + 3 * b ≤ 11) 
  (h2 : 3 * a + 6 * b ≤ 12) : 
  a + b ≤ 23 / 9 :=
sorry

end max_sum_with_constraints_l1240_124067


namespace jim_total_miles_l1240_124098

/-- Represents Jim's running schedule over 90 days -/
structure RunningSchedule where
  first_month : Nat  -- Miles per day for the first 30 days
  second_month : Nat -- Miles per day for the second 30 days
  third_month : Nat  -- Miles per day for the third 30 days

/-- Calculates the total miles run given a RunningSchedule -/
def total_miles (schedule : RunningSchedule) : Nat :=
  30 * schedule.first_month + 30 * schedule.second_month + 30 * schedule.third_month

/-- Theorem stating that Jim's total miles run is 1050 -/
theorem jim_total_miles :
  let jim_schedule : RunningSchedule := { first_month := 5, second_month := 10, third_month := 20 }
  total_miles jim_schedule = 1050 := by
  sorry


end jim_total_miles_l1240_124098


namespace consistency_comparison_l1240_124035

/-- Represents a player's performance in a basketball competition -/
structure PlayerPerformance where
  average_score : ℝ
  standard_deviation : ℝ

/-- Determines if a player performed more consistently than another -/
def more_consistent (p1 p2 : PlayerPerformance) : Prop :=
  p1.average_score = p2.average_score ∧ p1.standard_deviation < p2.standard_deviation

/-- Theorem: Given two players with the same average score, 
    the player with the smaller standard deviation performed more consistently -/
theorem consistency_comparison 
  (player_A player_B : PlayerPerformance) 
  (h_avg : player_A.average_score = player_B.average_score) 
  (h_std : player_B.standard_deviation < player_A.standard_deviation) : 
  more_consistent player_B player_A :=
sorry

end consistency_comparison_l1240_124035


namespace complex_modulus_equality_not_implies_square_equality_l1240_124032

theorem complex_modulus_equality_not_implies_square_equality :
  ∃ (z₁ z₂ : ℂ), Complex.abs z₁ = Complex.abs z₂ ∧ z₁^2 ≠ z₂^2 := by
  sorry

end complex_modulus_equality_not_implies_square_equality_l1240_124032


namespace imaginary_part_of_z_l1240_124011

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = -3 + 2 * Complex.I) :
  z.im = 3 := by
  sorry

end imaginary_part_of_z_l1240_124011


namespace vector_BC_l1240_124009

def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (4, 5)

theorem vector_BC : (C.1 - B.1, C.2 - B.2) = (3, 3) := by
  sorry

end vector_BC_l1240_124009


namespace min_value_when_a_is_one_exactly_two_zeros_l1240_124087

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x - a else 4*(x-a)*(x-2*a)

-- Theorem 1: Minimum value when a = 1
theorem min_value_when_a_is_one :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f 1 x_min ≤ f 1 x ∧ f 1 x_min = -1 :=
sorry

-- Theorem 2: Condition for exactly two zeros
theorem exactly_two_zeros (a : ℝ) :
  (∃ (x y : ℝ), x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ (z : ℝ), f a z = 0 → z = x ∨ z = y) ↔
  (1/2 ≤ a ∧ a < 1) ∨ (a ≥ 2) :=
sorry

end min_value_when_a_is_one_exactly_two_zeros_l1240_124087


namespace original_price_after_discount_l1240_124078

/-- Given a product with an unknown original price that becomes 50 yuan cheaper after a 20% discount, prove that its original price is 250 yuan. -/
theorem original_price_after_discount (price : ℝ) : 
  price * (1 - 0.2) = price - 50 → price = 250 := by
  sorry

end original_price_after_discount_l1240_124078


namespace speedster_convertibles_l1240_124055

theorem speedster_convertibles (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) : 
  (3 * speedsters = total) →  -- 1/3 of total inventory is Speedsters
  (5 * convertibles = 4 * speedsters) →  -- 4/5 of Speedsters are convertibles
  (total - speedsters = 30) →  -- 30 vehicles are not Speedsters
  convertibles = 12 := by
sorry

end speedster_convertibles_l1240_124055


namespace laptop_sale_price_l1240_124039

theorem laptop_sale_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 600 ∧ discount1 = 0.25 ∧ discount2 = 0.10 →
  (original_price * (1 - discount1) * (1 - discount2)) / original_price = 0.675 := by
sorry

end laptop_sale_price_l1240_124039


namespace store_purchase_combinations_l1240_124031

/-- The number of ways to buy three items (headphones, a keyboard, and a mouse) in a store with given inventory. -/
theorem store_purchase_combinations (headphones : ℕ) (mice : ℕ) (keyboards : ℕ) (keyboard_mouse_sets : ℕ) (headphone_mouse_sets : ℕ) : 
  headphones = 9 → 
  mice = 13 → 
  keyboards = 5 → 
  keyboard_mouse_sets = 4 → 
  headphone_mouse_sets = 5 → 
  headphones * keyboard_mouse_sets + 
  keyboards * headphone_mouse_sets + 
  headphones * mice * keyboards = 646 := by
sorry

end store_purchase_combinations_l1240_124031


namespace hyperbola_eccentricity_l1240_124046

-- Define the hyperbola C
def hyperbola_C : Set (ℝ × ℝ) := sorry

-- Define the foci of the hyperbola
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point P on both the hyperbola and the parabola
def P : ℝ × ℝ := sorry

-- Define the eccentricity of a hyperbola
def eccentricity (h : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := sorry

-- Define vector addition
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define vector subtraction
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := sorry

theorem hyperbola_eccentricity :
  P ∈ hyperbola_C ∧ 
  parabola P.1 P.2 ∧
  dot_product (vector_add (vector_sub P F₂) (vector_sub F₁ F₂)) 
              (vector_sub (vector_sub P F₂) (vector_sub F₁ F₂)) = 0 →
  eccentricity hyperbola_C = 1 + Real.sqrt 2 := by
  sorry

end hyperbola_eccentricity_l1240_124046


namespace lcm_of_9_12_15_l1240_124064

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end lcm_of_9_12_15_l1240_124064


namespace circle_line_relationship_l1240_124010

-- Define the circle C: x^2 + y^2 = 4
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l: √3x + y - 8 = 0
def l (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 8 = 0

-- Theorem statement
theorem circle_line_relationship :
  -- C and l are separate
  (∀ x y : ℝ, C x y → ¬ l x y) ∧
  -- The shortest distance from any point on C to l is 2
  (∀ x y : ℝ, C x y → ∃ d : ℝ, d = 2 ∧ 
    ∀ x' y' : ℝ, l x' y' → Real.sqrt ((x - x')^2 + (y - y')^2) ≥ d) :=
sorry

end circle_line_relationship_l1240_124010


namespace weather_forecast_inaccuracy_l1240_124096

theorem weather_forecast_inaccuracy (p_a p_b : ℝ) 
  (h_a : p_a = 0.9) 
  (h_b : p_b = 0.6) 
  (h_independent : True) -- Representing independence
  : (1 - p_a) * (1 - p_b) = 0.04 := by
  sorry

end weather_forecast_inaccuracy_l1240_124096


namespace composite_polynomial_l1240_124068

theorem composite_polynomial (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^3 + 6*n^2 + 12*n + 7 = a * b :=
by
  -- The proof would go here
  sorry

end composite_polynomial_l1240_124068


namespace five_students_three_communities_l1240_124061

/-- The number of ways to assign students to communities -/
def assign_students (n : ℕ) (k : ℕ) : ℕ :=
  -- Number of ways to assign n students to k communities
  -- with at least 1 student in each community
  sorry

/-- Theorem: 5 students assigned to 3 communities results in 150 ways -/
theorem five_students_three_communities :
  assign_students 5 3 = 150 := by
  sorry

end five_students_three_communities_l1240_124061


namespace arithmetic_mean_of_fractions_l1240_124074

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67/144 := by
  sorry

end arithmetic_mean_of_fractions_l1240_124074


namespace hcf_problem_l1240_124059

theorem hcf_problem (a b : ℕ) (h1 : a = 588) (h2 : a ≥ b) 
  (h3 : ∃ (hcf : ℕ), Nat.lcm a b = hcf * 12 * 14) : 
  Nat.gcd a b = 7 := by
  sorry

end hcf_problem_l1240_124059


namespace A_intersect_B_eq_open_interval_l1240_124094

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {x | Real.log x / Real.log 2 > Real.log x / Real.log 3}

-- Theorem statement
theorem A_intersect_B_eq_open_interval :
  A ∩ B = {x | 1 < x ∧ x < 3} := by sorry

end A_intersect_B_eq_open_interval_l1240_124094


namespace intersecting_lines_k_value_l1240_124017

/-- Given three lines that intersect at the same point, prove that k = -5 --/
theorem intersecting_lines_k_value (x y k : ℝ) :
  (y = 5 * x + 3) ∧
  (y = -2 * x - 25) ∧
  (y = 3 * x + k) →
  k = -5 := by
  sorry

end intersecting_lines_k_value_l1240_124017


namespace max_radius_circle_in_quartic_region_l1240_124082

/-- The maximum radius of a circle touching the origin and lying in y ≥ x^4 -/
theorem max_radius_circle_in_quartic_region : ∃ r : ℝ,
  (∀ x y : ℝ, x^2 + (y - r)^2 = r^2 → y ≥ x^4) ∧
  (∀ s : ℝ, s > r → ∃ x y : ℝ, x^2 + (y - s)^2 = s^2 ∧ y < x^4) ∧
  r = (3 * Real.rpow 2 (1/3 : ℝ)) / 4 :=
sorry

end max_radius_circle_in_quartic_region_l1240_124082


namespace area_triangle_ABP_l1240_124008

/-- Given points A and B in ℝ², and a point P on the x-axis forming a right angle with AB, 
    prove that the area of triangle ABP is 5/2. -/
theorem area_triangle_ABP (A B P : ℝ × ℝ) : 
  A = (1, 1) →
  B = (2, -1) →
  P.2 = 0 →  -- P is on the x-axis
  (P.1 - B.1) * (B.1 - A.1) + (P.2 - B.2) * (B.2 - A.2) = 0 →  -- ∠ABP = 90°
  abs ((A.1 - P.1) * (B.2 - P.2) - (B.1 - P.1) * (A.2 - P.2)) / 2 = 5/2 := by
sorry


end area_triangle_ABP_l1240_124008


namespace lemons_given_away_fraction_l1240_124058

def dozen : ℕ := 12

theorem lemons_given_away_fraction (lemons_left : ℕ) 
  (h1 : lemons_left = 9) : 
  (dozen - lemons_left : ℚ) / dozen = 1 / 4 := by
  sorry

end lemons_given_away_fraction_l1240_124058


namespace bag_balls_problem_l1240_124027

theorem bag_balls_problem (b g : ℕ) (p : ℚ) : 
  b = 8 →
  p = 1/3 →
  p = b / (b + g) →
  g = 16 :=
by
  sorry

end bag_balls_problem_l1240_124027


namespace inequality_condition_l1240_124048

theorem inequality_condition (t : ℝ) : (t + 1) * (1 - |t|) > 0 ↔ t < 1 ∧ t ≠ -1 := by
  sorry

end inequality_condition_l1240_124048


namespace complex_fraction_equality_l1240_124026

theorem complex_fraction_equality (z : ℂ) :
  z = 2 + I →
  (2 * I) / (z - 1) = 1 + I :=
by sorry

end complex_fraction_equality_l1240_124026


namespace second_investment_interest_rate_l1240_124018

theorem second_investment_interest_rate
  (total_income : ℝ)
  (investment1_principal : ℝ)
  (investment1_rate : ℝ)
  (investment2_principal : ℝ)
  (total_investment : ℝ)
  (h1 : total_income = 575)
  (h2 : investment1_principal = 3000)
  (h3 : investment1_rate = 0.085)
  (h4 : investment2_principal = 5000)
  (h5 : total_investment = 8000)
  (h6 : total_investment = investment1_principal + investment2_principal) :
  let investment1_income := investment1_principal * investment1_rate
  let investment2_income := total_income - investment1_income
  let investment2_rate := investment2_income / investment2_principal
  investment2_rate = 0.064 := by
  sorry

end second_investment_interest_rate_l1240_124018


namespace f_properties_l1240_124052

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  let k : ℤ := ⌊(x + 1) / 2⌋
  (-1: ℝ) ^ k * Real.sqrt (1 - (x - 2 * ↑k) ^ 2)

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, f (x + 4) = f x) ∧
  (∀ x : ℝ, f (x + 2) + f x = 0) := by
  sorry

end f_properties_l1240_124052


namespace medal_award_ways_eq_78_l1240_124080

/-- The number of ways to award medals in a race with American and non-American sprinters. -/
def medalAwardWays (totalSprinters : ℕ) (americanSprinters : ℕ) : ℕ :=
  let nonAmericanSprinters := totalSprinters - americanSprinters
  let noAmericanWins := nonAmericanSprinters * (nonAmericanSprinters - 1)
  let oneAmericanWins := 2 * americanSprinters * nonAmericanSprinters
  noAmericanWins + oneAmericanWins

/-- Theorem stating that the number of ways to award medals in the given scenario is 78. -/
theorem medal_award_ways_eq_78 :
  medalAwardWays 10 4 = 78 := by
  sorry

end medal_award_ways_eq_78_l1240_124080


namespace square_sum_squares_l1240_124056

theorem square_sum_squares (n : ℕ) : n < 200 → (∃ k : ℕ, n^2 + (n+1)^2 = k^2) ↔ n = 3 ∨ n = 20 ∨ n = 119 := by
  sorry

end square_sum_squares_l1240_124056


namespace cos_120_degrees_l1240_124085

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -(1 / 2) := by
  sorry

end cos_120_degrees_l1240_124085


namespace worker_b_time_l1240_124042

/-- Given two workers A and B, where A takes 8 hours to complete a job,
    and together they take 4.8 hours, prove that B takes 12 hours alone. -/
theorem worker_b_time (time_a time_ab : ℝ) (time_a_pos : time_a > 0) (time_ab_pos : time_ab > 0)
  (h1 : time_a = 8) (h2 : time_ab = 4.8) : 
  ∃ time_b : ℝ, time_b > 0 ∧ 1 / time_a + 1 / time_b = 1 / time_ab ∧ time_b = 12 := by
  sorry

#check worker_b_time

end worker_b_time_l1240_124042


namespace spherical_caps_ratio_l1240_124044

/-- 
Given a sphere of radius 1 cut by a plane into two spherical caps, 
if the combined surface area of the caps is 25% greater than the 
surface area of the original sphere, then the ratio of the surface 
areas of the larger cap to the smaller cap is (5 + 2√2) : (5 - 2√2).
-/
theorem spherical_caps_ratio (m₁ m₂ : ℝ) (ρ : ℝ) : 
  (0 < m₁) → (0 < m₂) → (0 < ρ) →
  (m₁ + m₂ = 2) →
  (2 * π * m₁ + π * ρ^2 + 2 * π * m₂ + π * ρ^2 = 5 * π) →
  (ρ^2 = 1 - (1 - m₁)^2) →
  (ρ^2 = 1 - (1 - m₂)^2) →
  ((2 * π * m₁ + π * ρ^2) / (2 * π * m₂ + π * ρ^2) = (5 + 2 * Real.sqrt 2) / (5 - 2 * Real.sqrt 2)) :=
by sorry

end spherical_caps_ratio_l1240_124044


namespace polynomial_remainder_l1240_124004

def polynomial (x : ℝ) : ℝ := 5*x^4 - 12*x^3 + 3*x^2 - 5*x + 15

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 108 := by
  sorry

end polynomial_remainder_l1240_124004


namespace right_triangle_area_l1240_124024

theorem right_triangle_area (a c : ℝ) (h1 : a = 30) (h2 : c = 34) : 
  let b := Real.sqrt (c^2 - a^2)
  (1/2) * a * b = 240 := by sorry

end right_triangle_area_l1240_124024


namespace square_pentagon_intersections_l1240_124072

/-- A square inscribed in a circle -/
structure InscribedSquare :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- A regular pentagon inscribed in a circle -/
structure InscribedPentagon :=
  (vertices : Fin 5 → ℝ × ℝ)

/-- Predicate to check if two polygons share a vertex -/
def ShareVertex (s : InscribedSquare) (p : InscribedPentagon) : Prop :=
  ∃ (i : Fin 4) (j : Fin 5), s.vertices i = p.vertices j

/-- The number of intersections between two polygons -/
def NumIntersections (s : InscribedSquare) (p : InscribedPentagon) : ℕ := sorry

/-- Theorem stating that a square and a regular pentagon inscribed in the same circle,
    not sharing any vertices, intersect at exactly 8 points -/
theorem square_pentagon_intersections
  (s : InscribedSquare) (p : InscribedPentagon)
  (h : ¬ ShareVertex s p) :
  NumIntersections s p = 8 :=
sorry

end square_pentagon_intersections_l1240_124072


namespace omega_range_l1240_124083

theorem omega_range (ω : ℝ) (h_pos : ω > 0) : 
  (∀ x ∈ Set.Ioo (2 * Real.pi / 3) (4 * Real.pi / 3), 
    Monotone (fun x => Real.cos (ω * x + Real.pi / 3))) → 
  1 ≤ ω ∧ ω ≤ 5 / 4 := by
  sorry

end omega_range_l1240_124083


namespace rectangle_perimeter_l1240_124049

theorem rectangle_perimeter (length width : ℝ) 
  (h1 : length * width = 360)
  (h2 : (length + 10) * (width - 6) = 360) :
  2 * (length + width) = 76 := by
  sorry

end rectangle_perimeter_l1240_124049


namespace constant_angle_existence_l1240_124053

-- Define the circle C
def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define the line L
def Line (a b c : ℝ) := {P : ℝ × ℝ | a * P.1 + b * P.2 + c = 0}

-- Define the condition that L does not intersect C
def DoesNotIntersect (C : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) := C ∩ L = ∅

-- Define the circle with diameter MN
def CircleWithDiameter (M N : ℝ × ℝ) := 
  Circle ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ((M.1 - N.1)^2 + (M.2 - N.2)^2)

-- Define the condition that CircleWithDiameter touches C but does not contain it
def TouchesButNotContains (C D : Set (ℝ × ℝ)) := 
  (∃ P, P ∈ C ∧ P ∈ D) ∧ (¬∃ P, P ∈ C ∧ P ∈ interior D)

-- Define the angle MPN
def Angle (M P N : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem constant_angle_existence 
  (C : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) (O : ℝ × ℝ) (r : ℝ) 
  (hC : C = Circle O r) (hL : ∃ a b c, L = Line a b c) 
  (hNotIntersect : DoesNotIntersect C L) :
  ∃ P : ℝ × ℝ, ∀ M N : ℝ × ℝ, 
    M ∈ L → N ∈ L → 
    TouchesButNotContains C (CircleWithDiameter M N) →
    ∃ θ : ℝ, Angle M P N = θ :=
sorry

end constant_angle_existence_l1240_124053


namespace max_value_of_f_l1240_124081

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1 / x else -x^2 + 2

theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end max_value_of_f_l1240_124081


namespace paddle_prices_and_cost_effective_solution_l1240_124070

/-- Represents the price of a pair of straight paddles in yuan -/
def straight_paddle_price : ℝ := sorry

/-- Represents the price of a pair of horizontal paddles in yuan -/
def horizontal_paddle_price : ℝ := sorry

/-- Cost of table tennis balls per pair of paddles -/
def ball_cost : ℝ := 20

/-- Total cost for 20 pairs of straight paddles and 15 pairs of horizontal paddles -/
def total_cost_35_pairs : ℝ := 9000

/-- Difference in cost between 10 pairs of horizontal paddles and 5 pairs of straight paddles -/
def cost_difference : ℝ := 1600

/-- Theorem stating the prices of paddles and the cost-effective solution -/
theorem paddle_prices_and_cost_effective_solution :
  (straight_paddle_price = 220 ∧ horizontal_paddle_price = 260) ∧
  (∀ m : ℕ, m ≤ 40 → m ≤ 3 * (40 - m) →
    m * (straight_paddle_price + ball_cost) + (40 - m) * (horizontal_paddle_price + ball_cost) ≥ 10000) ∧
  (30 * (straight_paddle_price + ball_cost) + 10 * (horizontal_paddle_price + ball_cost) = 10000) :=
by sorry

end paddle_prices_and_cost_effective_solution_l1240_124070


namespace car_overtake_distance_l1240_124063

/-- Represents the distance between two cars -/
def distance_between_cars (v1 v2 t : ℝ) : ℝ := (v2 - v1) * t

/-- Theorem stating the distance between two cars under given conditions -/
theorem car_overtake_distance :
  let red_speed : ℝ := 30
  let black_speed : ℝ := 50
  let overtake_time : ℝ := 1
  distance_between_cars red_speed black_speed overtake_time = 20 := by
  sorry

end car_overtake_distance_l1240_124063


namespace pattern_equation_l1240_124051

theorem pattern_equation (n : ℕ) (h : n ≥ 2) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) := by
  sorry

end pattern_equation_l1240_124051


namespace average_marks_combined_classes_l1240_124022

theorem average_marks_combined_classes (students1 : ℕ) (students2 : ℕ) (avg1 : ℝ) (avg2 : ℝ) :
  students1 = 25 →
  students2 = 40 →
  avg1 = 50 →
  avg2 = 65 →
  let total_students := students1 + students2
  let total_marks := students1 * avg1 + students2 * avg2
  abs ((total_marks / total_students) - 59.23) < 0.01 := by
  sorry

end average_marks_combined_classes_l1240_124022


namespace symmetric_points_l1240_124006

/-- Given that point A(2, 4) is symmetric to point B(b-1, 2a) with respect to the origin, prove that a - b = -1 -/
theorem symmetric_points (a b : ℝ) : 
  (2 = -(b - 1) ∧ 4 = -2*a) → a - b = -1 := by
  sorry

end symmetric_points_l1240_124006


namespace points_on_decreasing_line_l1240_124047

theorem points_on_decreasing_line (a₁ a₂ b₁ b₂ : ℝ) :
  a₁ ≠ a₂ →
  b₁ = -3 * a₁ + 4 →
  b₂ = -3 * a₂ + 4 →
  (a₁ - a₂) * (b₁ - b₂) < 0 :=
by sorry

end points_on_decreasing_line_l1240_124047


namespace max_a_value_l1240_124023

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, a * x^2 + 2 * a * x + 3 * a ≤ 1) →
  a ≤ 1/6 :=
sorry

end max_a_value_l1240_124023


namespace alice_walked_distance_l1240_124054

/-- The distance Alice walked in miles -/
def alice_distance (blocks_south : ℕ) (blocks_west : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_south + blocks_west : ℚ) * miles_per_block

/-- Theorem stating that Alice walked 3.25 miles -/
theorem alice_walked_distance :
  alice_distance 5 8 (1/4) = 3.25 := by sorry

end alice_walked_distance_l1240_124054


namespace min_value_cube_root_plus_inverse_square_l1240_124057

theorem min_value_cube_root_plus_inverse_square (x : ℝ) (h : x > 0) :
  3 * x^(1/3) + 4 / x^2 ≥ 7 ∧
  (3 * x^(1/3) + 4 / x^2 = 7 ↔ x = 1) :=
sorry

end min_value_cube_root_plus_inverse_square_l1240_124057


namespace h_expansion_count_h_expansion_10_l1240_124045

/-- Definition of H expansion sequence -/
def h_expansion_seq (n : ℕ) : ℕ :=
  2^n + 1

/-- Theorem: The number of items after n H expansions is 2^n + 1 -/
theorem h_expansion_count (n : ℕ) :
  h_expansion_seq n = 2^n + 1 :=
by sorry

/-- Corollary: After 10 H expansions, the sequence has 1025 items -/
theorem h_expansion_10 :
  h_expansion_seq 10 = 1025 :=
by sorry

end h_expansion_count_h_expansion_10_l1240_124045


namespace fifth_term_of_geometric_sequence_l1240_124021

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 8) :
  a 5 = 16 :=
sorry

end fifth_term_of_geometric_sequence_l1240_124021


namespace geometric_sequence_second_term_l1240_124092

/-- A geometric sequence with third term 5 and fifth term 45 has 5/3 as a possible second term -/
theorem geometric_sequence_second_term (a r : ℝ) : 
  a * r^2 = 5 → a * r^4 = 45 → a * r = 5/3 ∨ a * r = -5/3 :=
by sorry

end geometric_sequence_second_term_l1240_124092


namespace friend_walking_rates_l1240_124001

theorem friend_walking_rates (trail_length : ℝ) (p_distance : ℝ) 
  (h1 : trail_length = 36)
  (h2 : p_distance = 20)
  (h3 : p_distance < trail_length) :
  let q_distance := trail_length - p_distance
  let rate_ratio := p_distance / q_distance
  (rate_ratio - 1) * 100 = 25 := by
  sorry

end friend_walking_rates_l1240_124001


namespace polynomial_equation_l1240_124036

/-- Given polynomials h and p such that h(x) + p(x) = 3x^2 - x + 4 
    and h(x) = x^4 - 5x^2 + x + 6, prove that p(x) = -x^4 + 8x^2 - 2x - 2 -/
theorem polynomial_equation (x : ℝ) (h p : ℝ → ℝ) 
    (h_p_sum : ∀ x, h x + p x = 3 * x^2 - x + 4)
    (h_def : ∀ x, h x = x^4 - 5 * x^2 + x + 6) :
  p x = -x^4 + 8 * x^2 - 2 * x - 2 := by
  sorry

end polynomial_equation_l1240_124036


namespace unique_solution_quadratic_inequality_l1240_124020

theorem unique_solution_quadratic_inequality (m : ℝ) : 
  (∃! x : ℝ, x^2 - m*x + 1 ≤ 0) → (m = 2 ∨ m = -2) :=
sorry

end unique_solution_quadratic_inequality_l1240_124020


namespace final_amount_is_correct_l1240_124077

/-- The final amount owed after applying three consecutive 5% late charges to an initial bill of $200. -/
def final_amount : ℝ := 200 * (1.05)^3

/-- Theorem stating that the final amount owed is $231.525 -/
theorem final_amount_is_correct : final_amount = 231.525 := by
  sorry

end final_amount_is_correct_l1240_124077


namespace min_value_f_inequality_abc_l1240_124041

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - 2)

-- Theorem for the minimum value of f(x)
theorem min_value_f : ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

-- Theorem for the inequality
theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  b^2 / a + c^2 / b + a^2 / c ≥ 3 :=
sorry

end min_value_f_inequality_abc_l1240_124041


namespace estimate_wheat_amount_l1240_124028

/-- Estimates the amount of wheat in a mixed batch of grain -/
theorem estimate_wheat_amount (total_mixed : ℕ) (sample_size : ℕ) (wheat_in_sample : ℕ) : 
  total_mixed = 1524 →
  sample_size = 254 →
  wheat_in_sample = 28 →
  (total_mixed * wheat_in_sample) / sample_size = 168 :=
by sorry

end estimate_wheat_amount_l1240_124028


namespace max_sum_of_coeff_bound_l1240_124086

/-- A complex polynomial of degree 2 -/
def ComplexPoly (a b c : ℂ) : ℂ → ℂ := fun z ↦ a * z^2 + b * z + c

/-- The statement that |f(z)| ≤ 1 for all |z| ≤ 1 -/
def BoundedOnUnitDisk (f : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs z ≤ 1 → Complex.abs (f z) ≤ 1

/-- The main theorem -/
theorem max_sum_of_coeff_bound {a b c : ℂ} (h : BoundedOnUnitDisk (ComplexPoly a b c)) :
    Complex.abs a + Complex.abs b ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

#check max_sum_of_coeff_bound

end max_sum_of_coeff_bound_l1240_124086


namespace expand_product_l1240_124038

theorem expand_product (x : ℝ) : (2 + x^2) * (3 - x^3 + x^5) = 6 + 3*x^2 - 2*x^3 + x^5 + x^7 := by
  sorry

end expand_product_l1240_124038


namespace chord_length_l1240_124062

/-- Given a circle and a line intersecting at two points, 
    prove that the length of the chord formed by these intersection points is 9√5 / 5 -/
theorem chord_length (x y : ℝ) : 
  (x^2 + y^2 + 4*x - 4*y - 10 = 0) →  -- Circle equation
  (2*x - y + 1 = 0) →                -- Line equation
  ∃ (A B : ℝ × ℝ),                   -- Existence of intersection points A and B
    (A.1^2 + A.2^2 + 4*A.1 - 4*A.2 - 10 = 0) ∧ 
    (2*A.1 - A.2 + 1 = 0) ∧
    (B.1^2 + B.2^2 + 4*B.1 - 4*B.2 - 10 = 0) ∧ 
    (2*B.1 - B.2 + 1 = 0) ∧
    (A ≠ B) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (9*Real.sqrt 5 / 5)^2) := by
  sorry

end chord_length_l1240_124062
