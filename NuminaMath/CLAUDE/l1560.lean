import Mathlib

namespace school_survey_sampling_params_l1560_156089

/-- Systematic sampling parameters for a given population and sample size -/
def systematic_sampling_params (population : ℕ) (sample_size : ℕ) : ℕ × ℕ :=
  let n := population % sample_size
  let m := sample_size
  (n, m)

/-- Theorem stating the correct systematic sampling parameters for the given problem -/
theorem school_survey_sampling_params :
  systematic_sampling_params 1553 50 = (3, 50) := by
sorry

end school_survey_sampling_params_l1560_156089


namespace february_messages_l1560_156062

def text_messages (month : ℕ) : ℕ :=
  2^month

theorem february_messages :
  text_messages 3 = 8 ∧ text_messages 4 = 16 :=
by sorry

end february_messages_l1560_156062


namespace penny_species_count_l1560_156054

theorem penny_species_count :
  let sharks : ℕ := 35
  let eels : ℕ := 15
  let whales : ℕ := 5
  sharks + eels + whales = 55 := by
  sorry

end penny_species_count_l1560_156054


namespace diamonds_in_G_15_l1560_156008

-- Define the sequence G
def G : ℕ → ℕ
| 0 => 1  -- G_1 has 1 diamond
| n + 1 => G n + 4 * (n + 2)  -- G_{n+1} adds 4 sides with (n+2) more diamonds each

-- Theorem statement
theorem diamonds_in_G_15 : G 14 = 1849 := by
  sorry

end diamonds_in_G_15_l1560_156008


namespace area_of_overlapping_squares_l1560_156063

/-- Represents a square in a 2D plane -/
structure Square where
  sideLength : ℝ
  center : ℝ × ℝ

/-- Calculates the area of the region covered by two overlapping squares -/
def areaCoveredByOverlappingSquares (s1 s2 : Square) : ℝ :=
  sorry

/-- Theorem stating the area covered by two specific overlapping squares -/
theorem area_of_overlapping_squares :
  let s1 := Square.mk 12 (0, 0)
  let s2 := Square.mk 12 (6, 6)
  areaCoveredByOverlappingSquares s1 s2 = 144 := by
  sorry

end area_of_overlapping_squares_l1560_156063


namespace is_center_of_symmetry_l1560_156071

/-- The function f(x) = (x+2)³ - x + 1 -/
def f (x : ℝ) : ℝ := (x + 2)^3 - x + 1

/-- The center of symmetry for the function f -/
def center_of_symmetry : ℝ × ℝ := (-2, 3)

/-- Theorem stating that the given point is the center of symmetry for f -/
theorem is_center_of_symmetry :
  ∀ x : ℝ, f (center_of_symmetry.1 + x) + f (center_of_symmetry.1 - x) = 2 * center_of_symmetry.2 :=
sorry

end is_center_of_symmetry_l1560_156071


namespace area_of_polygon_l1560_156041

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a polygon -/
structure Polygon :=
  (vertices : List Point)

/-- Calculates the area of a polygon -/
def area (p : Polygon) : ℝ := sorry

/-- Checks if a quadrilateral is a rectangle -/
def is_rectangle (a b c d : Point) : Prop := sorry

/-- Checks if a quadrilateral is a square -/
def is_square (a b f e : Point) : Prop := sorry

/-- Checks if two line segments are perpendicular -/
def is_perpendicular (a f : Point) (f e : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

theorem area_of_polygon (a b c d e f : Point) :
  is_rectangle a b c d →
  is_square a b f e →
  is_perpendicular a f f e →
  distance a f = 10 →
  distance f e = 15 →
  distance c d = 20 →
  area (Polygon.mk [a, f, e, d, c, b]) = 375 := by
  sorry

end area_of_polygon_l1560_156041


namespace sum_of_four_sqrt_inequality_l1560_156093

theorem sum_of_four_sqrt_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) :
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) + Real.sqrt (4 * d + 1) < 6 := by
  sorry

end sum_of_four_sqrt_inequality_l1560_156093


namespace car_final_velocity_l1560_156002

/-- Calculates the final velocity of a car parallel to the ground after accelerating on an inclined slope. -/
theorem car_final_velocity (u : Real) (a : Real) (t : Real) (θ : Real) :
  u = 10 ∧ a = 2 ∧ t = 3 ∧ θ = 15 * π / 180 →
  ∃ v : Real, abs (v - (u + a * t) * Real.cos θ) < 0.0001 ∧ abs (v - 15.4544) < 0.0001 :=
by sorry

end car_final_velocity_l1560_156002


namespace iggys_pace_l1560_156083

/-- Iggy's running schedule for the week -/
def daily_miles : List Nat := [3, 4, 6, 8, 3]

/-- Total time Iggy spent running in hours -/
def total_hours : Nat := 4

/-- Calculate Iggy's pace in minutes per mile -/
def calculate_pace (miles : List Nat) (hours : Nat) : Nat :=
  let total_miles := miles.sum
  let total_minutes := hours * 60
  total_minutes / total_miles

/-- Theorem: Iggy's pace is 10 minutes per mile -/
theorem iggys_pace :
  calculate_pace daily_miles total_hours = 10 := by
  sorry

end iggys_pace_l1560_156083


namespace salad_cost_main_theorem_l1560_156078

/-- The cost of ingredients for Laura's dinner --/
structure DinnerCost where
  salad_price : ℝ
  beef_price : ℝ
  potato_price : ℝ
  juice_price : ℝ

/-- The quantities of ingredients Laura bought --/
structure DinnerQuantities where
  salad_qty : ℕ
  beef_qty : ℕ
  potato_qty : ℕ
  juice_qty : ℕ

/-- The theorem stating the cost of one salad --/
theorem salad_cost (d : DinnerCost) (q : DinnerQuantities) : d.salad_price = 3 :=
  by
    have h1 : d.beef_price = 2 * d.salad_price := sorry
    have h2 : d.potato_price = (1/3) * d.salad_price := sorry
    have h3 : d.juice_price = 1.5 := sorry
    have h4 : q.salad_qty = 2 ∧ q.beef_qty = 2 ∧ q.potato_qty = 1 ∧ q.juice_qty = 2 := sorry
    have h5 : q.salad_qty * d.salad_price + q.beef_qty * d.beef_price + 
              q.potato_qty * d.potato_price + q.juice_qty * d.juice_price = 22 := sorry
    sorry

/-- The main theorem proving the cost of one salad --/
theorem main_theorem : ∃ (d : DinnerCost) (q : DinnerQuantities), d.salad_price = 3 :=
  by
    sorry

end salad_cost_main_theorem_l1560_156078


namespace solve_exponential_equation_l1560_156098

theorem solve_exponential_equation :
  ∃ x : ℝ, (5 : ℝ) ^ (2 * x + 3) = 125 ^ (x + 1) ∧ x = 0 := by
sorry

end solve_exponential_equation_l1560_156098


namespace new_oranges_added_l1560_156027

/-- Calculates the number of new oranges added to a bin -/
def new_oranges (initial : ℕ) (thrown_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - thrown_away)

/-- Proves that the number of new oranges added is 28 -/
theorem new_oranges_added : new_oranges 5 2 31 = 28 := by
  sorry

end new_oranges_added_l1560_156027


namespace g_five_times_one_l1560_156081

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x + 2 else 3 * x + 1

theorem g_five_times_one : g (g (g (g (g 1)))) = 12 := by
  sorry

end g_five_times_one_l1560_156081


namespace school_gender_difference_l1560_156065

theorem school_gender_difference 
  (initial_girls : ℕ) 
  (initial_boys : ℕ) 
  (additional_girls : ℕ) 
  (h1 : initial_girls = 632)
  (h2 : initial_boys = 410)
  (h3 : additional_girls = 465) :
  initial_girls + additional_girls - initial_boys = 687 := by
  sorry

end school_gender_difference_l1560_156065


namespace f_less_than_4_iff_in_M_abs_sum_less_than_abs_product_plus_4_l1560_156021

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Define the set M
def M : Set ℝ := Set.Ioo (-2) 2

-- Statement 1
theorem f_less_than_4_iff_in_M : ∀ x : ℝ, f x < 4 ↔ x ∈ M := by sorry

-- Statement 2
theorem abs_sum_less_than_abs_product_plus_4 : 
  ∀ x y : ℝ, x ∈ M → y ∈ M → |x + y| < |x * y / 2 + 2| := by sorry

end f_less_than_4_iff_in_M_abs_sum_less_than_abs_product_plus_4_l1560_156021


namespace circumscribed_circle_area_l1560_156067

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units -/
theorem circumscribed_circle_area (s : ℝ) (h : s = 12) : 
  let r := 2 * s * Real.sqrt 3 / 3
  (π : ℝ) * r^2 = 48 * π := by sorry

end circumscribed_circle_area_l1560_156067


namespace no_single_digit_divisor_l1560_156018

theorem no_single_digit_divisor (n : ℤ) (d : ℤ) :
  1 < d → d < 10 → ¬(∃ k : ℤ, 2 * n^2 - 31 = d * k) := by
  sorry

end no_single_digit_divisor_l1560_156018


namespace camel_cost_l1560_156073

/-- The cost of animals in a zoo -/
structure AnimalCosts where
  camel : ℕ
  horse : ℕ
  ox : ℕ
  elephant : ℕ
  giraffe : ℕ
  zebra : ℕ

/-- The conditions given in the problem -/
def zoo_conditions (c : AnimalCosts) : Prop :=
  10 * c.camel = 24 * c.horse ∧
  16 * c.horse = 4 * c.ox ∧
  6 * c.ox = 4 * c.elephant ∧
  3 * c.elephant = 15 * c.giraffe ∧
  8 * c.giraffe = 20 * c.zebra ∧
  12 * c.elephant = 180000

theorem camel_cost (c : AnimalCosts) :
  zoo_conditions c → c.camel = 6000 := by
  sorry

end camel_cost_l1560_156073


namespace polynomial_division_quotient_l1560_156099

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 10 * X^4 + 5 * X^3 - 8 * X^2 + 7 * X - 3
  let divisor : Polynomial ℚ := 3 * X + 2
  let quotient : Polynomial ℚ := (10/3) * X^3 - (5/9) * X^2 - (31/27) * X + 143/81
  (dividend.div divisor = quotient) ∧ (dividend.mod divisor).degree < divisor.degree :=
by sorry

end polynomial_division_quotient_l1560_156099


namespace taxi_initial_fee_l1560_156085

/-- Represents the taxi service charging model -/
structure TaxiCharge where
  initialFee : ℝ
  additionalChargePerSegment : ℝ
  segmentLength : ℝ
  totalDistance : ℝ
  totalCharge : ℝ

/-- Theorem: Given the taxi service charging model, prove that the initial fee is $2.25 -/
theorem taxi_initial_fee (t : TaxiCharge) : 
  t.additionalChargePerSegment = 0.3 ∧ 
  t.segmentLength = 2/5 ∧ 
  t.totalDistance = 3.6 ∧ 
  t.totalCharge = 4.95 → 
  t.initialFee = 2.25 := by
  sorry

end taxi_initial_fee_l1560_156085


namespace largest_four_digit_base4_is_255_l1560_156034

/-- Converts a base-4 digit to its base-10 equivalent -/
def base4DigitToBase10 (digit : Nat) : Nat :=
  if digit < 4 then digit else 0

/-- Calculates the base-10 value of a four-digit base-4 number -/
def fourDigitBase4ToBase10 (d1 d2 d3 d4 : Nat) : Nat :=
  (base4DigitToBase10 d1) * (4^3) +
  (base4DigitToBase10 d2) * (4^2) +
  (base4DigitToBase10 d3) * (4^1) +
  (base4DigitToBase10 d4) * (4^0)

/-- The largest four-digit base-4 number, when converted to base-10, equals 255 -/
theorem largest_four_digit_base4_is_255 :
  fourDigitBase4ToBase10 3 3 3 3 = 255 := by
  sorry

#eval fourDigitBase4ToBase10 3 3 3 3

end largest_four_digit_base4_is_255_l1560_156034


namespace inequality_proof_l1560_156091

theorem inequality_proof (a b c d e f : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f) :
  Real.rpow (a * b * c / (a + b + d)) (1/3) + Real.rpow (d * e * f / (c + e + f)) (1/3) 
  < Real.rpow ((a + b + d) * (c + e + f)) (1/3) := by
  sorry

end inequality_proof_l1560_156091


namespace f_inequality_solution_set_l1560_156095

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + m - 1

-- State the theorem
theorem f_inequality_solution_set (m : ℝ) :
  (∀ x : ℝ, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

end f_inequality_solution_set_l1560_156095


namespace angle_magnification_l1560_156043

theorem angle_magnification (original_angle : ℝ) (magnification : ℝ) :
  original_angle = 20 ∧ magnification = 10 →
  original_angle = original_angle := by sorry

end angle_magnification_l1560_156043


namespace inverse_variation_y_sqrt_z_l1560_156042

theorem inverse_variation_y_sqrt_z (y z : ℝ) (k : ℝ) (h1 : y^2 * Real.sqrt z = k) 
  (h2 : 3^2 * Real.sqrt 4 = k) (h3 : y = 6) : z = 1/4 := by
  sorry

end inverse_variation_y_sqrt_z_l1560_156042


namespace square_difference_l1560_156024

theorem square_difference (m n : ℕ+) 
  (h : (2001 : ℕ) * m ^ 2 + m = (2002 : ℕ) * n ^ 2 + n) :
  ∃ k : ℕ, (m : ℤ) - (n : ℤ) = k ^ 2 := by
  sorry

end square_difference_l1560_156024


namespace book_price_problem_l1560_156019

theorem book_price_problem (n : ℕ) (a : ℕ → ℝ) :
  n = 41 ∧
  a 1 = 7 ∧
  (∀ i, 1 ≤ i ∧ i < n → a (i + 1) = a i + 3) ∧
  a n = a ((n + 1) / 2) + a (((n + 1) / 2) + 1) →
  a ((n + 1) / 2) = 67 := by
sorry

end book_price_problem_l1560_156019


namespace quadratic_inequality_solution_set_l1560_156016

theorem quadratic_inequality_solution_set (a : ℝ) :
  {x : ℝ | x^2 - (2*a + 1)*x + a^2 + a < 0} = Set.Ioo a (a + 1) := by
  sorry

end quadratic_inequality_solution_set_l1560_156016


namespace hours_per_day_is_five_l1560_156022

/-- The number of hours worked per day by the first group of women -/
def hours_per_day : ℝ := 5

/-- The number of women in the first group -/
def women_group1 : ℕ := 6

/-- The number of days worked by the first group -/
def days_group1 : ℕ := 8

/-- The units of work completed by the first group -/
def work_units_group1 : ℕ := 75

/-- The number of women in the second group -/
def women_group2 : ℕ := 4

/-- The number of days worked by the second group -/
def days_group2 : ℕ := 3

/-- The units of work completed by the second group -/
def work_units_group2 : ℕ := 30

/-- The number of hours worked per day by the second group -/
def hours_per_day_group2 : ℕ := 8

/-- The proposition that the amount of work done is proportional to the number of woman-hours worked -/
axiom work_proportional_to_hours : 
  (women_group1 * days_group1 * hours_per_day) / work_units_group1 = 
  (women_group2 * days_group2 * hours_per_day_group2) / work_units_group2

theorem hours_per_day_is_five : hours_per_day = 5 := by
  sorry

end hours_per_day_is_five_l1560_156022


namespace water_level_rise_l1560_156037

-- Define the cube's edge length
def cube_edge : ℝ := 16

-- Define the vessel's base dimensions
def vessel_length : ℝ := 20
def vessel_width : ℝ := 15

-- Define the volume of the cube
def cube_volume : ℝ := cube_edge ^ 3

-- Define the area of the vessel's base
def vessel_base_area : ℝ := vessel_length * vessel_width

-- Theorem statement
theorem water_level_rise :
  (cube_volume / vessel_base_area) = (cube_edge ^ 3) / (vessel_length * vessel_width) :=
by sorry

end water_level_rise_l1560_156037


namespace triangle_two_solutions_l1560_156092

-- Define the triangle
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0

-- State the theorem
theorem triangle_two_solutions 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_a : a = 2)
  (h_A : A = Real.pi / 3) -- 60° in radians
  (h_two_solutions : b * Real.sin A < a ∧ a < b) :
  2 < b ∧ b < 4 * Real.sqrt 3 / 3 :=
sorry

end triangle_two_solutions_l1560_156092


namespace least_frood_count_l1560_156009

/-- The function representing points earned by dropping n froods -/
def drop_points (n : ℕ) : ℚ := n * (n + 1) / 2

/-- The function representing points earned by eating n froods -/
def eat_points (n : ℕ) : ℚ := 20 * n

/-- The theorem stating that 40 is the least positive integer for which
    dropping froods earns more points than eating them -/
theorem least_frood_count : ∀ n : ℕ, n > 0 → (drop_points n > eat_points n ↔ n ≥ 40) := by
  sorry

end least_frood_count_l1560_156009


namespace hyperbola_focal_distance_l1560_156088

/-- The hyperbola with equation x^2/16 - y^2/20 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 16 - p.2^2 / 20 = 1}

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- Distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_focal_distance 
  (P : ℝ × ℝ) 
  (h_P : P ∈ Hyperbola) 
  (h_dist : distance P F₁ = 9) : 
  distance P F₂ = 17 := by sorry

end hyperbola_focal_distance_l1560_156088


namespace median_and_mode_are_23_l1560_156082

/-- Represents the shoe size distribution of a class --/
structure ShoeSizeDistribution where
  sizes : List Nat
  frequencies : List Nat
  total_students : Nat
  h_sizes_freq : sizes.length = frequencies.length
  h_total : total_students = frequencies.sum

/-- Calculates the median of a shoe size distribution --/
def median (d : ShoeSizeDistribution) : Nat :=
  sorry

/-- Calculates the mode of a shoe size distribution --/
def mode (d : ShoeSizeDistribution) : Nat :=
  sorry

/-- The shoe size distribution for the class in the problem --/
def class_distribution : ShoeSizeDistribution :=
  { sizes := [20, 21, 22, 23, 24],
    frequencies := [2, 8, 9, 19, 2],
    total_students := 40,
    h_sizes_freq := by rfl,
    h_total := by rfl }

theorem median_and_mode_are_23 :
  median class_distribution = 23 ∧ mode class_distribution = 23 :=
sorry

end median_and_mode_are_23_l1560_156082


namespace train_pass_man_time_l1560_156004

def train_speed : Real := 36 -- km/hr
def platform_length : Real := 180 -- meters
def time_pass_platform : Real := 30 -- seconds

theorem train_pass_man_time : 
  ∃ (train_length : Real),
    (train_speed * 1000 / 3600 * time_pass_platform = train_length + platform_length) ∧
    (train_length / (train_speed * 1000 / 3600) = 12) :=
by sorry

end train_pass_man_time_l1560_156004


namespace one_french_horn_player_l1560_156076

/-- Represents the number of players for each instrument in an orchestra -/
structure Orchestra :=
  (total : ℕ)
  (drummer : ℕ)
  (trombone : ℕ)
  (trumpet : ℕ)
  (violin : ℕ)
  (cello : ℕ)
  (contrabass : ℕ)
  (clarinet : ℕ)
  (flute : ℕ)
  (maestro : ℕ)

/-- Theorem stating that there is one French horn player in the orchestra -/
theorem one_french_horn_player (o : Orchestra) 
  (h_total : o.total = 21)
  (h_drummer : o.drummer = 1)
  (h_trombone : o.trombone = 4)
  (h_trumpet : o.trumpet = 2)
  (h_violin : o.violin = 3)
  (h_cello : o.cello = 1)
  (h_contrabass : o.contrabass = 1)
  (h_clarinet : o.clarinet = 3)
  (h_flute : o.flute = 4)
  (h_maestro : o.maestro = 1) :
  o.total = o.drummer + o.trombone + o.trumpet + o.violin + o.cello + o.contrabass + o.clarinet + o.flute + o.maestro + 1 :=
by sorry

end one_french_horn_player_l1560_156076


namespace journey_speed_l1560_156097

/-- Given a journey with total distance D and total time T, prove that if a person
    travels 2/3 of D in 1/3 of T at 40 kmph, they must travel at 10 kmph for the
    remaining distance to arrive on time. -/
theorem journey_speed (D T : ℝ) (h_positive : D > 0 ∧ T > 0) : 
  (2/3 * D) / (1/3 * T) = 40 → (1/3 * D) / (2/3 * T) = 10 := by
  sorry

end journey_speed_l1560_156097


namespace min_coach_handshakes_l1560_156057

/-- The total number of handshakes at the gymnastics meet -/
def total_handshakes : ℕ := 903

/-- The number of gymnasts at the meet -/
def n : ℕ := 43

/-- The number of coaches at the meet -/
def num_coaches : ℕ := 3

/-- Function to calculate the number of handshakes between gymnasts -/
def gymnast_handshakes (m : ℕ) : ℕ := m * (m - 1) / 2

/-- Theorem stating the minimum number of handshakes involving coaches -/
theorem min_coach_handshakes : 
  ∃ (k₁ k₂ k₃ : ℕ), 
    gymnast_handshakes n + k₁ + k₂ + k₃ = total_handshakes ∧ 
    k₁ + k₂ + k₃ = 0 := by
  sorry

end min_coach_handshakes_l1560_156057


namespace full_price_revenue_is_2128_l1560_156080

/-- Represents the ticket sale scenario -/
structure TicketSale where
  total_tickets : ℕ
  total_revenue : ℕ
  full_price_tickets : ℕ
  discounted_tickets : ℕ
  full_price : ℕ

/-- The conditions of the ticket sale -/
def valid_ticket_sale (sale : TicketSale) : Prop :=
  sale.total_tickets = 200 ∧
  sale.total_revenue = 2688 ∧
  sale.full_price_tickets + sale.discounted_tickets = sale.total_tickets ∧
  sale.full_price_tickets * sale.full_price + sale.discounted_tickets * (sale.full_price / 3) = sale.total_revenue

/-- The theorem to be proved -/
theorem full_price_revenue_is_2128 (sale : TicketSale) :
  valid_ticket_sale sale →
  sale.full_price_tickets * sale.full_price = 2128 :=
by sorry

end full_price_revenue_is_2128_l1560_156080


namespace max_correct_answers_l1560_156059

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (blank_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 50 →
  correct_points = 4 →
  blank_points = 0 →
  incorrect_points = -1 →
  total_score = 99 →
  ∃ (max_correct : ℕ), 
    max_correct ≤ total_questions ∧
    (∀ (correct blank incorrect : ℕ),
      correct + blank + incorrect = total_questions →
      correct_points * correct + blank_points * blank + incorrect_points * incorrect = total_score →
      correct ≤ max_correct) ∧
    max_correct = 29 :=
by sorry

end max_correct_answers_l1560_156059


namespace problem_statement_l1560_156020

theorem problem_statement :
  (∃ x : ℝ, x^2 - x + 1 ≥ 0) ∧
  ¬(∀ a b : ℝ, a^2 < b^2 → a < b) :=
by sorry

end problem_statement_l1560_156020


namespace base_six_addition_l1560_156007

/-- Given a base 6 addition problem 3XY_6 + 23_6 = 41X_6, prove that X + Y = 7 in base 10 -/
theorem base_six_addition (X Y : ℕ) : 
  (3 * 6^2 + X * 6 + Y) + (2 * 6 + 3) = 4 * 6^2 + X * 6 → X + Y = 7 :=
by sorry

end base_six_addition_l1560_156007


namespace canoe_downstream_speed_l1560_156025

/-- The speed of a canoe downstream given its upstream speed and the stream speed -/
theorem canoe_downstream_speed (upstream_speed stream_speed : ℝ) :
  upstream_speed = 3 →
  stream_speed = 4.5 →
  upstream_speed + 2 * stream_speed = 12 :=
by sorry

end canoe_downstream_speed_l1560_156025


namespace john_bought_three_puzzles_l1560_156029

/-- Represents the number of puzzles John bought -/
def num_puzzles : ℕ := 3

/-- Represents the number of pieces in the first puzzle -/
def first_puzzle_pieces : ℕ := 1000

/-- Represents the number of pieces in the second and third puzzles -/
def other_puzzle_pieces : ℕ := (3 * first_puzzle_pieces) / 2

/-- The total number of pieces in all puzzles -/
def total_pieces : ℕ := 4000

/-- Theorem stating that the number of puzzles John bought is 3 -/
theorem john_bought_three_puzzles :
  num_puzzles = 3 ∧
  first_puzzle_pieces = 1000 ∧
  other_puzzle_pieces = (3 * first_puzzle_pieces) / 2 ∧
  total_pieces = first_puzzle_pieces + 2 * other_puzzle_pieces :=
by sorry

end john_bought_three_puzzles_l1560_156029


namespace triangle_angle_calculation_l1560_156056

theorem triangle_angle_calculation (a b : ℝ) (A B : ℝ) :
  a = Real.sqrt 3 * b →
  A = 2 * π / 3 →
  B = π / 6 :=
by
  sorry

end triangle_angle_calculation_l1560_156056


namespace min_value_theorem_l1560_156094

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1 / x^2 + 1 / y + 1 / z = 6) :
  x^3 * y^2 * z^2 ≥ 1 / (8 * Real.sqrt 2) ∧
  ∃ x₀ y₀ z₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    1 / x₀^2 + 1 / y₀ + 1 / z₀ = 6 ∧
    x₀^3 * y₀^2 * z₀^2 = 1 / (8 * Real.sqrt 2) :=
by sorry

end min_value_theorem_l1560_156094


namespace expansion_coefficient_l1560_156055

/-- The coefficient of x^3 in the expansion of ((ax-1)^6) -/
def coefficient_x3 (a : ℝ) : ℝ := -20 * a^3

/-- The theorem states that if the coefficient of x^3 in the expansion of ((ax-1)^6) is 20, then a = -1 -/
theorem expansion_coefficient (a : ℝ) : coefficient_x3 a = 20 → a = -1 := by
  sorry

end expansion_coefficient_l1560_156055


namespace large_cheese_block_volume_l1560_156075

/-- Represents the dimensions and volume of a cheese block -/
structure CheeseBlock where
  width : ℝ
  depth : ℝ
  length : ℝ
  volume : ℝ

/-- Theorem: Volume of a large cheese block -/
theorem large_cheese_block_volume
  (normal : CheeseBlock)
  (large : CheeseBlock)
  (h1 : normal.volume = 3)
  (h2 : large.width = 2 * normal.width)
  (h3 : large.depth = 2 * normal.depth)
  (h4 : large.length = 3 * normal.length)
  (h5 : large.volume = large.width * large.depth * large.length) :
  large.volume = 36 := by
  sorry

#check large_cheese_block_volume

end large_cheese_block_volume_l1560_156075


namespace tan_difference_l1560_156011

theorem tan_difference (α β : Real) 
  (h1 : Real.tan (α + β) = 1/2) 
  (h2 : Real.tan (α + Real.pi/4) = -1/3) : 
  Real.tan (β - Real.pi/4) = 1 := by
  sorry

end tan_difference_l1560_156011


namespace xy_value_l1560_156047

theorem xy_value (x y : ℝ) (h : (x^2 + 6*x + 12) * (5*y^2 + 2*y + 1) = 12/5) : 
  x * y = 3/5 := by
  sorry

end xy_value_l1560_156047


namespace exists_permutation_with_difference_l1560_156038

theorem exists_permutation_with_difference (x y z w : ℝ) 
  (sum_eq : x + y + z + w = 13)
  (sum_squares_eq : x^2 + y^2 + z^2 + w^2 = 43) :
  ∃ (a b c d : ℝ), ({a, b, c, d} : Finset ℝ) = {x, y, z, w} ∧ a * b - c * d ≥ 3 := by
  sorry

end exists_permutation_with_difference_l1560_156038


namespace min_dot_product_l1560_156000

/-- A line with direction vector (4, -4) passing through (0, -4) -/
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (t, -t - 4)}

/-- Two points on line_l -/
def point_on_line (M N : ℝ × ℝ) : Prop :=
  M ∈ line_l ∧ N ∈ line_l

/-- Distance between two points is 4 -/
def distance_is_4 (M N : ℝ × ℝ) : Prop :=
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 16

/-- Dot product of OM and ON -/
def dot_product (M N : ℝ × ℝ) : ℝ :=
  M.1 * N.1 + M.2 * N.2

theorem min_dot_product (M N : ℝ × ℝ) 
  (h1 : point_on_line M N) 
  (h2 : distance_is_4 M N) : 
  ∃ min_val : ℝ, min_val = 4 ∧ ∀ M' N' : ℝ × ℝ, 
    point_on_line M' N' → distance_is_4 M' N' → 
    dot_product M' N' ≥ min_val :=
  sorry

end min_dot_product_l1560_156000


namespace print_shop_charge_difference_l1560_156015

/-- The charge per color copy at print shop X -/
def charge_x : ℚ := 125/100

/-- The charge per color copy at print shop Y -/
def charge_y : ℚ := 275/100

/-- The number of color copies -/
def num_copies : ℕ := 40

/-- The difference in charges between print shop Y and X for num_copies color copies -/
def charge_difference : ℚ := num_copies * charge_y - num_copies * charge_x

theorem print_shop_charge_difference : charge_difference = 60 := by
  sorry

end print_shop_charge_difference_l1560_156015


namespace sheila_tue_thu_hours_l1560_156031

/-- Represents Sheila's work schedule and earnings -/
structure WorkSchedule where
  hoursPerDayMWF : ℕ  -- Hours worked on Monday, Wednesday, Friday
  daysWorkedMWF : ℕ   -- Number of days worked (Monday, Wednesday, Friday)
  weeklyEarnings : ℕ  -- Total earnings per week
  hourlyRate : ℕ      -- Hourly rate of pay

/-- Calculates the total hours worked on Tuesday and Thursday -/
def hoursTueThu (schedule : WorkSchedule) : ℕ :=
  let mwfHours := schedule.hoursPerDayMWF * schedule.daysWorkedMWF
  let mwfEarnings := mwfHours * schedule.hourlyRate
  let tueThuEarnings := schedule.weeklyEarnings - mwfEarnings
  tueThuEarnings / schedule.hourlyRate

/-- Theorem: Given Sheila's work schedule, she works 12 hours on Tuesday and Thursday combined -/
theorem sheila_tue_thu_hours (schedule : WorkSchedule) 
  (h1 : schedule.hoursPerDayMWF = 8)
  (h2 : schedule.daysWorkedMWF = 3)
  (h3 : schedule.weeklyEarnings = 360)
  (h4 : schedule.hourlyRate = 10) :
  hoursTueThu schedule = 12 := by
  sorry

#eval hoursTueThu { hoursPerDayMWF := 8, daysWorkedMWF := 3, weeklyEarnings := 360, hourlyRate := 10 }

end sheila_tue_thu_hours_l1560_156031


namespace infinitely_many_larger_divisor_sum_ratio_l1560_156066

-- Define the sum of divisors function
def sigma (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem infinitely_many_larger_divisor_sum_ratio :
  ∀ t : ℕ, ∃ n : ℕ, n > t ∧ ∀ k : ℕ, k ∈ Finset.range n → (sigma n : ℚ) / n > (sigma k : ℚ) / k :=
sorry

end infinitely_many_larger_divisor_sum_ratio_l1560_156066


namespace james_toys_problem_l1560_156084

theorem james_toys_problem (sell_percentage : Real) (buy_price : Real) (sell_price : Real) (total_profit : Real) :
  sell_percentage = 0.8 →
  buy_price = 20 →
  sell_price = 30 →
  total_profit = 800 →
  ∃ initial_toys : Real, initial_toys = 100 ∧ 
    sell_percentage * initial_toys * (sell_price - buy_price) = total_profit := by
  sorry

end james_toys_problem_l1560_156084


namespace square_completion_l1560_156090

theorem square_completion (x : ℝ) : x^2 + 5*x + 25/4 = (x + 5/2)^2 := by
  sorry

end square_completion_l1560_156090


namespace no_real_roots_l1560_156017

theorem no_real_roots : ∀ x : ℝ, (x + 1) * |x + 1| - x * |x| + 1 ≠ 0 := by
  sorry

end no_real_roots_l1560_156017


namespace vector_problem_l1560_156044

def a : ℝ × ℝ := (1, 3)
def b (y : ℝ) : ℝ × ℝ := (2, y)

theorem vector_problem (y : ℝ) :
  (∀ y, (a.1 * (b y).1 + a.2 * (b y).2 = 5) → y = 1) ∧
  (∀ y, ((a.1 + (b y).1)^2 + (a.2 + (b y).2)^2 = (a.1 - (b y).1)^2 + (a.2 - (b y).2)^2) → y = -2/3) :=
by sorry

end vector_problem_l1560_156044


namespace burger_cost_l1560_156052

theorem burger_cost : ∃ (burger_cost : ℝ),
  burger_cost = 9 ∧
  ∃ (pizza_cost : ℝ),
  pizza_cost = 2 * burger_cost ∧
  pizza_cost + 3 * burger_cost = 45 := by
sorry

end burger_cost_l1560_156052


namespace collinear_probability_is_1_182_l1560_156070

/-- Represents a 4x4 square array of dots -/
def SquareArray : Type := Fin 4 × Fin 4

/-- The total number of dots in the array -/
def totalDots : Nat := 16

/-- The number of ways to choose 4 dots from the array -/
def totalChoices : Nat := Nat.choose totalDots 4

/-- The number of sets of 4 collinear dots in the array -/
def collinearSets : Nat := 10

/-- The probability of choosing 4 collinear dots -/
def collinearProbability : Rat := collinearSets / totalChoices

theorem collinear_probability_is_1_182 :
  collinearProbability = 1 / 182 := by sorry

end collinear_probability_is_1_182_l1560_156070


namespace gasoline_price_increase_l1560_156058

/-- The percentage increase in gasoline price from 1972 to 1992 -/
theorem gasoline_price_increase (initial_price final_price : ℝ) : 
  initial_price = 29.90 →
  final_price = 149.70 →
  (final_price - initial_price) / initial_price * 100 = 400 := by
sorry

end gasoline_price_increase_l1560_156058


namespace largest_c_value_l1560_156060

theorem largest_c_value (c : ℝ) (h : (3 * c + 4) * (c - 2) = 9 * c) : 
  ∀ x : ℝ, (3 * x + 4) * (x - 2) = 9 * x → x ≤ 4 := by sorry

end largest_c_value_l1560_156060


namespace sams_weight_l1560_156086

/-- Given the weights of Tyler, Sam, and Peter, prove Sam's weight -/
theorem sams_weight (tyler sam peter : ℝ) : 
  tyler = sam + 25 →
  peter = tyler / 2 →
  peter = 65 →
  sam = 105 := by
  sorry

end sams_weight_l1560_156086


namespace decimal_multiplication_l1560_156023

theorem decimal_multiplication (h : 28 * 15 = 420) :
  (2.8 * 1.5 = 4.2) ∧ (0.28 * 1.5 = 42) ∧ (0.028 * 0.15 = 0.0042) := by
  sorry

end decimal_multiplication_l1560_156023


namespace yellow_balls_count_l1560_156064

theorem yellow_balls_count (total : ℕ) (white : ℕ) (green : ℕ) (red : ℕ) (purple : ℕ) (prob : ℚ) :
  total = 100 →
  white = 50 →
  green = 20 →
  red = 17 →
  purple = 3 →
  prob = 4/5 →
  (white + green + (total - white - green - red - purple) : ℚ) / total = prob →
  total - white - green - red - purple = 10 :=
by sorry

end yellow_balls_count_l1560_156064


namespace vessel_width_calculation_l1560_156005

-- Define the given parameters
def cube_edge : ℝ := 15
def vessel_length : ℝ := 20
def water_rise : ℝ := 12.053571428571429

-- Define the theorem
theorem vessel_width_calculation (w : ℝ) :
  (cube_edge ^ 3 = vessel_length * w * water_rise) →
  w = 14 := by
  sorry

end vessel_width_calculation_l1560_156005


namespace angle_C_measure_triangle_area_l1560_156032

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.c = 2 ∧ 2 * Real.sin t.A = Real.sqrt 3 * t.a * Real.cos t.C

theorem angle_C_measure (t : Triangle) (h : TriangleConditions t) : 
  t.C = π / 3 := by sorry

theorem triangle_area (t : Triangle) (h : TriangleConditions t) 
  (h2 : 2 * Real.sin (2 * t.A) + Real.sin (2 * t.B + t.C) = Real.sin t.C) : 
  (1/2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3 / 3 := by sorry

end angle_C_measure_triangle_area_l1560_156032


namespace exists_equitable_non_symmetric_polygon_l1560_156087

-- Define a polygon as a list of points in 2D space
def Polygon := List (ℝ × ℝ)

-- Function to check if a polygon has no self-intersections
def hasNoSelfIntersections (p : Polygon) : Prop :=
  sorry

-- Function to check if a line through the origin divides a polygon into two regions of equal area
def dividesEquallyThroughOrigin (p : Polygon) (line : ℝ → ℝ) : Prop :=
  sorry

-- Function to check if a polygon is equitable
def isEquitable (p : Polygon) : Prop :=
  ∀ line : ℝ → ℝ, dividesEquallyThroughOrigin p line

-- Function to check if a polygon is centrally symmetric about the origin
def isCentrallySymmetric (p : Polygon) : Prop :=
  sorry

-- Theorem statement
theorem exists_equitable_non_symmetric_polygon :
  ∃ p : Polygon, hasNoSelfIntersections p ∧ isEquitable p ∧ ¬(isCentrallySymmetric p) :=
sorry

end exists_equitable_non_symmetric_polygon_l1560_156087


namespace specific_conference_handshakes_l1560_156053

/-- The number of handshakes in a conference with gremlins and imps -/
def conference_handshakes (num_gremlins num_imps : ℕ) : ℕ :=
  let gremlin_gremlin := num_gremlins.choose 2
  let gremlin_imp := num_gremlins * num_imps
  gremlin_gremlin + gremlin_imp

/-- Theorem stating the number of handshakes in the specific conference -/
theorem specific_conference_handshakes :
  conference_handshakes 25 10 = 550 := by
  sorry

end specific_conference_handshakes_l1560_156053


namespace trig_identity_l1560_156077

theorem trig_identity : 
  Real.sin (-1071 * π / 180) * Real.sin (99 * π / 180) + 
  Real.sin (-171 * π / 180) * Real.sin (-261 * π / 180) + 
  Real.tan (-1089 * π / 180) * Real.tan (-540 * π / 180) = 0 := by
  sorry

end trig_identity_l1560_156077


namespace water_amount_from_reaction_l1560_156012

-- Define the chemical species
inductive ChemicalSpecies
| NaOH
| HClO4
| NaClO4
| H2O

-- Define the reaction equation
def reactionEquation : List (ChemicalSpecies × ℕ) := 
  [(ChemicalSpecies.NaOH, 1), (ChemicalSpecies.HClO4, 1), 
   (ChemicalSpecies.NaClO4, 1), (ChemicalSpecies.H2O, 1)]

-- Define the molar mass of water
def molarMassWater : ℝ := 18.015

-- Define the amount of reactants
def amountNaOH : ℝ := 1
def amountHClO4 : ℝ := 1

-- Theorem statement
theorem water_amount_from_reaction :
  let waterFormed := amountNaOH * molarMassWater
  waterFormed = 18.015 := by sorry

end water_amount_from_reaction_l1560_156012


namespace batsman_average_is_59_l1560_156048

/-- Calculates the batting average given the total innings, highest score, 
    average excluding highest and lowest scores, and the difference between highest and lowest scores. -/
def battingAverage (totalInnings : ℕ) (highestScore : ℕ) (averageExcludingExtremes : ℕ) (scoreDifference : ℕ) : ℚ :=
  let lowestScore := highestScore - scoreDifference
  let totalScore := (totalInnings - 2) * averageExcludingExtremes + highestScore + lowestScore
  totalScore / totalInnings

/-- Theorem stating that under the given conditions, the batting average is 59 runs. -/
theorem batsman_average_is_59 :
  battingAverage 46 156 58 150 = 59 := by sorry

end batsman_average_is_59_l1560_156048


namespace cycle_original_price_l1560_156039

/-- Given a cycle sold at a 10% loss for Rs. 1080, prove its original price was Rs. 1200 -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) :
  selling_price = 1080 →
  loss_percentage = 10 →
  selling_price = (1 - loss_percentage / 100) * 1200 :=
by sorry

end cycle_original_price_l1560_156039


namespace flower_color_difference_l1560_156096

/-- Given the following flower counts:
  - Total flowers: 60
  - Yellow and white flowers: 13
  - Red and yellow flowers: 17
  - Red and white flowers: 14
  - Blue and yellow flowers: 16

  Prove that there are 4 more flowers containing red than white. -/
theorem flower_color_difference
  (total : ℕ)
  (yellow_white : ℕ)
  (red_yellow : ℕ)
  (red_white : ℕ)
  (blue_yellow : ℕ)
  (h_total : total = 60)
  (h_yellow_white : yellow_white = 13)
  (h_red_yellow : red_yellow = 17)
  (h_red_white : red_white = 14)
  (h_blue_yellow : blue_yellow = 16) :
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by sorry

end flower_color_difference_l1560_156096


namespace triangle_angle_relation_l1560_156061

theorem triangle_angle_relation (X Y Z Z₁ Z₂ : ℝ) : 
  X = 40 → Y = 50 → X + Y + Z = 180 → Z = Z₁ + Z₂ → Z₁ - Z₂ = 10 := by
  sorry

end triangle_angle_relation_l1560_156061


namespace expression_simplification_l1560_156035

theorem expression_simplification (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end expression_simplification_l1560_156035


namespace bingo_prize_distribution_l1560_156050

theorem bingo_prize_distribution (total_prize : ℝ) (remaining_winners : ℕ) (each_remaining_prize : ℝ) 
  (h1 : total_prize = 2400)
  (h2 : remaining_winners = 10)
  (h3 : each_remaining_prize = 160)
  (h4 : ∀ f : ℝ, (1 - f) * total_prize / remaining_winners = each_remaining_prize → f = 1/3) :
  ∃ f : ℝ, f * total_prize = total_prize / 3 ∧ 
    (1 - f) * total_prize / remaining_winners = each_remaining_prize := by
  sorry

#check bingo_prize_distribution

end bingo_prize_distribution_l1560_156050


namespace donut_distribution_l1560_156006

/-- The number of ways to distribute n identical objects into k distinct boxes,
    with each box containing at least m objects. -/
def distributionWays (n k m : ℕ) : ℕ := sorry

/-- The theorem stating that there are 10 ways to distribute 10 donuts
    into 4 kinds with at least 2 of each kind. -/
theorem donut_distribution : distributionWays 10 4 2 = 10 := by sorry

end donut_distribution_l1560_156006


namespace triangle_perimeter_l1560_156026

/-- Given a triangle with inradius 2.5 cm and area 45 cm², its perimeter is 36 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 45 → A = r * (p / 2) → p = 36 := by
  sorry

end triangle_perimeter_l1560_156026


namespace rose_work_days_l1560_156045

/-- Given that Paul completes a work in 80 days and Paul and Rose together
    complete the same work in 48 days, prove that Rose completes the work
    alone in 120 days. -/
theorem rose_work_days (paul_days : ℕ) (together_days : ℕ) (rose_days : ℕ) : 
  paul_days = 80 → together_days = 48 → 
  1 / paul_days + 1 / rose_days = 1 / together_days →
  rose_days = 120 := by
  sorry

end rose_work_days_l1560_156045


namespace tom_weeds_earnings_l1560_156003

/-- Tom's lawn mowing business -/
def tom_lawn_business (weeds_earnings : ℕ) : Prop :=
  let lawns_mowed : ℕ := 3
  let charge_per_lawn : ℕ := 12
  let gas_cost : ℕ := 17
  let total_profit : ℕ := 29
  let mowing_profit : ℕ := lawns_mowed * charge_per_lawn - gas_cost
  weeds_earnings = total_profit - mowing_profit

theorem tom_weeds_earnings : 
  ∃ (weeds_earnings : ℕ), tom_lawn_business weeds_earnings ∧ weeds_earnings = 10 :=
sorry

end tom_weeds_earnings_l1560_156003


namespace tan_monotone_or_angle_sin_equivalence_l1560_156046

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- Define a predicate for monotonically increasing functions
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define a triangle
structure Triangle :=
  (A B C : ℝ)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- State the theorem
theorem tan_monotone_or_angle_sin_equivalence :
  (MonotonicallyIncreasing tan) ∨ 
  (∀ t : Triangle, t.A > t.B ↔ Real.sin t.A > Real.sin t.B) :=
sorry

end tan_monotone_or_angle_sin_equivalence_l1560_156046


namespace container_volume_scaling_l1560_156068

theorem container_volume_scaling (V k : ℝ) (h : k > 0) :
  let new_volume := V * k^3
  new_volume = V * k * k * k :=
by sorry

end container_volume_scaling_l1560_156068


namespace inequality_and_equality_conditions_l1560_156072

theorem inequality_and_equality_conditions (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 ∧
  ((a - b) * (b - c) * (a - c) = 2 ↔ 
    ((a = 0 ∧ b = 2 ∧ c = 1) ∨ 
     (a = 2 ∧ b = 1 ∧ c = 0) ∨ 
     (a = 1 ∧ b = 0 ∧ c = 2))) :=
by sorry

end inequality_and_equality_conditions_l1560_156072


namespace distance_A_to_y_axis_l1560_156001

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate -/
def distanceToYAxis (x : ℝ) (y : ℝ) : ℝ := |x|

/-- Point A has coordinates (2, -3) -/
def pointA : ℝ × ℝ := (2, -3)

/-- Theorem: The distance from point A(2, -3) to the y-axis is 2 -/
theorem distance_A_to_y_axis :
  distanceToYAxis pointA.1 pointA.2 = 2 := by sorry

end distance_A_to_y_axis_l1560_156001


namespace joseph_cards_l1560_156069

theorem joseph_cards (initial_cards : ℕ) (cards_to_friend : ℕ) (remaining_fraction : ℚ) : 
  initial_cards = 16 →
  cards_to_friend = 2 →
  remaining_fraction = 1/2 →
  (initial_cards - cards_to_friend - (remaining_fraction * initial_cards)) / initial_cards = 3/8 := by
sorry

end joseph_cards_l1560_156069


namespace right_triangle_sides_l1560_156010

theorem right_triangle_sides : ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  a = 5 → b = 12 → c = 13 →
  a^2 + b^2 = c^2 := by
  sorry

end right_triangle_sides_l1560_156010


namespace smallest_n_for_quadratic_inequality_six_satisfies_inequality_smallest_n_is_six_l1560_156030

theorem smallest_n_for_quadratic_inequality :
  ∀ n : ℤ, n^2 - 9*n + 20 > 0 → n ≥ 6 :=
by
  sorry

theorem six_satisfies_inequality : (6 : ℤ)^2 - 9*(6 : ℤ) + 20 > 0 :=
by
  sorry

theorem smallest_n_is_six :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 9*m + 20 > 0 → m ≥ n) ∧ n^2 - 9*n + 20 > 0 ∧ n = 6 :=
by
  sorry

end smallest_n_for_quadratic_inequality_six_satisfies_inequality_smallest_n_is_six_l1560_156030


namespace kevin_cards_total_l1560_156013

/-- Given that Kevin starts with 7 cards and finds 47 more, prove that he ends up with 54 cards in total. -/
theorem kevin_cards_total : 
  let initial_cards : ℕ := 7
  let found_cards : ℕ := 47
  initial_cards + found_cards = 54 := by sorry

end kevin_cards_total_l1560_156013


namespace min_value_3a_4b_l1560_156036

theorem min_value_3a_4b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : (a + b) * (a + 2 * b) + a + b = 9) : 
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ (x + y) * (x + 2 * y) + x + y = 9 → 
  3 * x + 4 * y ≥ 6 * Real.sqrt 2 - 1 :=
sorry

end min_value_3a_4b_l1560_156036


namespace smallest_base_perfect_square_l1560_156074

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 4 ∧ b = 6 ∧ ∃ (n : ℕ), 2 * b + 4 = n^2 ∧
  ∀ (c : ℕ), c > 4 ∧ c < b → ¬∃ (m : ℕ), 2 * c + 4 = m^2 :=
sorry

end smallest_base_perfect_square_l1560_156074


namespace cappuccino_cost_l1560_156028

theorem cappuccino_cost (cappuccino_cost : ℝ) : 
  (3 : ℝ) * cappuccino_cost + 2 * 3 + 2 * 1.5 + 2 * 1 = 20 - 3 → 
  cappuccino_cost = 2 := by
  sorry

end cappuccino_cost_l1560_156028


namespace profit_percentage_l1560_156051

theorem profit_percentage (selling_price cost_price : ℝ) :
  cost_price = 0.96 * selling_price →
  (selling_price - cost_price) / cost_price * 100 = (1 / 24) * 100 := by
sorry

end profit_percentage_l1560_156051


namespace polynomial_coefficient_sum_l1560_156033

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 6) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 0 := by
sorry

end polynomial_coefficient_sum_l1560_156033


namespace taxi_charge_proof_l1560_156040

/-- Calculates the total charge for a taxi trip -/
def total_charge (initial_fee : ℚ) (rate_per_increment : ℚ) (increment_distance : ℚ) (trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / increment_distance).floor * rate_per_increment

/-- Proves that the total charge for a 3.6-mile trip is $7.65 -/
theorem taxi_charge_proof :
  let initial_fee : ℚ := 9/4  -- $2.25
  let rate_per_increment : ℚ := 3/10  -- $0.3
  let increment_distance : ℚ := 2/5  -- 2/5 mile
  let trip_distance : ℚ := 18/5  -- 3.6 miles
  total_charge initial_fee rate_per_increment increment_distance trip_distance = 153/20  -- $7.65
  := by sorry


end taxi_charge_proof_l1560_156040


namespace largest_cube_forming_integer_l1560_156049

theorem largest_cube_forming_integer : 
  ∀ n : ℕ, n > 19 → ¬∃ k : ℤ, n^3 + 4*n^2 - 15*n - 18 = k^3 :=
by sorry

end largest_cube_forming_integer_l1560_156049


namespace trigonometric_simplification_l1560_156079

theorem trigonometric_simplification (α : ℝ) : 
  3.4113 * Real.sin α * Real.cos (3 * α) + 
  9 * Real.sin α * Real.cos α - 
  Real.sin (3 * α) * Real.cos (3 * α) - 
  3 * Real.sin (3 * α) * Real.cos α = 
  2 * (Real.sin (2 * α))^3 := by sorry

end trigonometric_simplification_l1560_156079


namespace unknown_number_proof_l1560_156014

theorem unknown_number_proof (x : ℝ) : 
  (10 + 30 + 50) / 3 = (20 + x + 6) / 3 + 8 → x = 40 := by
  sorry

end unknown_number_proof_l1560_156014
