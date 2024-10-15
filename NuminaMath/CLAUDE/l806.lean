import Mathlib

namespace NUMINAMATH_CALUDE_no_integer_regular_quadrilateral_pyramid_l806_80623

theorem no_integer_regular_quadrilateral_pyramid :
  ¬ ∃ (g h f s v : ℕ+),
    (f : ℝ)^2 = (h : ℝ)^2 + (g : ℝ)^2 / 2 ∧
    (s : ℝ) = (g : ℝ)^2 + 2 * (g : ℝ) * Real.sqrt ((h : ℝ)^2 + (g : ℝ)^2 / 4) ∧
    (v : ℝ) = (g : ℝ)^2 * (h : ℝ) / 3 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_regular_quadrilateral_pyramid_l806_80623


namespace NUMINAMATH_CALUDE_road_length_l806_80626

/-- Proves that a road with streetlights installed every 10 meters on both sides, 
    with a total of 120 streetlights, is 590 meters long. -/
theorem road_length (streetlight_interval : Nat) (total_streetlights : Nat) (road_length : Nat) : 
  streetlight_interval = 10 → 
  total_streetlights = 120 → 
  road_length = (total_streetlights / 2 - 1) * streetlight_interval → 
  road_length = 590 :=
by sorry

end NUMINAMATH_CALUDE_road_length_l806_80626


namespace NUMINAMATH_CALUDE_remainder_theorem_l806_80630

/-- The dividend polynomial -/
def f (x : ℝ) : ℝ := 3*x^5 - 2*x^3 + 5*x - 8

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- The proposed remainder -/
def r (x : ℝ) : ℝ := 84*x - 84

/-- Theorem stating that r is the remainder when f is divided by g -/
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l806_80630


namespace NUMINAMATH_CALUDE_feb_1_2015_was_sunday_l806_80647

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

/-- Theorem: If January 1, 2015 was a Thursday, then February 1, 2015 was a Sunday -/
theorem feb_1_2015_was_sunday :
  advanceDays DayOfWeek.Thursday 31 = DayOfWeek.Sunday := by
  sorry

end NUMINAMATH_CALUDE_feb_1_2015_was_sunday_l806_80647


namespace NUMINAMATH_CALUDE_parallel_statements_l806_80675

-- Define the concept of parallel lines
def parallel_lines (l1 l2 : Line) : Prop := sorry

-- Define the concept of parallel planes
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Define a line being parallel to a plane
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

theorem parallel_statements :
  -- Statement 1
  (∀ l1 l2 l3 : Line, parallel_lines l1 l3 → parallel_lines l2 l3 → parallel_lines l1 l2) ∧
  -- Statement 2
  (∀ p1 p2 p3 : Plane, parallel_planes p1 p3 → parallel_planes p2 p3 → parallel_planes p1 p2) ∧
  -- Statement 3 (negation)
  (∃ l1 l2 : Line, ∃ p : Plane, 
    parallel_lines l1 l2 ∧ line_parallel_to_plane l1 p ∧ ¬line_parallel_to_plane l2 p) ∧
  -- Statement 4 (negation)
  (∃ l : Line, ∃ p1 p2 : Plane,
    parallel_planes p1 p2 ∧ line_parallel_to_plane l p1 ∧ ¬line_parallel_to_plane l p2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_statements_l806_80675


namespace NUMINAMATH_CALUDE_functional_equation_properties_l806_80604

theorem functional_equation_properties 
  (f g h : ℝ → ℝ)
  (hf : ∀ x y : ℝ, f (x * y) = x * f y)
  (hg : ∀ x y : ℝ, g (x * y) = x * g y)
  (hh : ∀ x y : ℝ, h (x * y) = x * h y) :
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x y : ℝ, (g ∘ h) (x * y) = x * (g ∘ h) y) ∧
  (g ∘ h = h ∘ g) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l806_80604


namespace NUMINAMATH_CALUDE_pete_walked_7430_miles_l806_80629

/-- Represents a pedometer with a maximum value before flipping --/
structure Pedometer where
  max_value : ℕ
  flip_count : ℕ
  final_reading : ℕ

/-- Calculates the total steps recorded by a pedometer --/
def total_steps (p : Pedometer) : ℕ :=
  p.max_value * p.flip_count + p.final_reading

/-- Represents Pete's walking data for the year --/
structure WalkingData where
  pedometer1 : Pedometer
  pedometer2 : Pedometer
  steps_per_mile : ℕ

/-- Theorem stating that Pete walked 7430 miles during the year --/
theorem pete_walked_7430_miles (data : WalkingData)
  (h1 : data.pedometer1.max_value = 100000)
  (h2 : data.pedometer1.flip_count = 50)
  (h3 : data.pedometer1.final_reading = 25000)
  (h4 : data.pedometer2.max_value = 400000)
  (h5 : data.pedometer2.flip_count = 15)
  (h6 : data.pedometer2.final_reading = 120000)
  (h7 : data.steps_per_mile = 1500) :
  (total_steps data.pedometer1 + total_steps data.pedometer2) / data.steps_per_mile = 7430 := by
  sorry

end NUMINAMATH_CALUDE_pete_walked_7430_miles_l806_80629


namespace NUMINAMATH_CALUDE_jordan_born_in_1980_l806_80635

/-- The year when the first AMC 8 was given -/
def first_amc8_year : ℕ := 1985

/-- The age of Jordan when he took the tenth AMC 8 contest -/
def jordan_age_at_tenth_amc8 : ℕ := 14

/-- The number of years between the first AMC 8 and the tenth AMC 8 -/
def years_between_first_and_tenth : ℕ := 9

/-- Jordan's birth year -/
def jordan_birth_year : ℕ := first_amc8_year + years_between_first_and_tenth - jordan_age_at_tenth_amc8

theorem jordan_born_in_1980 : jordan_birth_year = 1980 := by
  sorry

end NUMINAMATH_CALUDE_jordan_born_in_1980_l806_80635


namespace NUMINAMATH_CALUDE_dandelion_puffs_count_l806_80689

/-- The number of dandelion puffs Caleb originally picked -/
def original_puffs : ℕ := 40

/-- The number of puffs given to mom -/
def mom_puffs : ℕ := 3

/-- The number of puffs given to sister -/
def sister_puffs : ℕ := 3

/-- The number of puffs given to grandmother -/
def grandmother_puffs : ℕ := 5

/-- The number of puffs given to dog -/
def dog_puffs : ℕ := 2

/-- The number of friends -/
def num_friends : ℕ := 3

/-- The number of puffs each friend received -/
def puffs_per_friend : ℕ := 9

theorem dandelion_puffs_count :
  original_puffs = mom_puffs + sister_puffs + grandmother_puffs + dog_puffs + num_friends * puffs_per_friend :=
by sorry

end NUMINAMATH_CALUDE_dandelion_puffs_count_l806_80689


namespace NUMINAMATH_CALUDE_sin_tan_40_deg_l806_80634

theorem sin_tan_40_deg : 4 * Real.sin (40 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_tan_40_deg_l806_80634


namespace NUMINAMATH_CALUDE_intersection_theorem_l806_80627

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 + 2*x < 3}

def B : Set ℝ := {x | (x-2)/x ≤ 0}

theorem intersection_theorem :
  A ∩ (U \ B) = {x : ℝ | -3 < x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l806_80627


namespace NUMINAMATH_CALUDE_hydropolis_aquaville_rainfall_difference_l806_80603

/-- The difference in total rainfall between two cities over a year, given their average monthly rainfalls and the number of months. -/
def rainfall_difference (avg_rainfall_city1 avg_rainfall_city2 : ℝ) (months : ℕ) : ℝ :=
  (avg_rainfall_city1 - avg_rainfall_city2) * months

/-- Theorem stating the difference in total rainfall between Hydropolis and Aquaville in 2011 -/
theorem hydropolis_aquaville_rainfall_difference :
  let hydropolis_2010 : ℝ := 36.5
  let rainfall_increase : ℝ := 3.5
  let hydropolis_2011 : ℝ := hydropolis_2010 + rainfall_increase
  let aquaville_2011 : ℝ := hydropolis_2011 - 1.5
  let months : ℕ := 12
  rainfall_difference hydropolis_2011 aquaville_2011 months = 18.0 := by
  sorry

#eval rainfall_difference 40.0 38.5 12

end NUMINAMATH_CALUDE_hydropolis_aquaville_rainfall_difference_l806_80603


namespace NUMINAMATH_CALUDE_julia_watch_collection_l806_80673

theorem julia_watch_collection (silver_watches : ℕ) (bronze_watches : ℕ) (gold_watches : ℕ) : 
  silver_watches = 20 →
  bronze_watches = 3 * silver_watches →
  gold_watches = (silver_watches + bronze_watches) / 10 →
  silver_watches + bronze_watches + gold_watches = 88 := by
  sorry

end NUMINAMATH_CALUDE_julia_watch_collection_l806_80673


namespace NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l806_80632

/-- A convex hexagon -/
structure ConvexHexagon where
  -- Add any necessary properties of a convex hexagon

/-- A diagonal of a convex hexagon -/
structure Diagonal (H : ConvexHexagon) where
  -- Add any necessary properties of a diagonal

/-- Two diagonals intersect inside the hexagon (not at a vertex) -/
def intersect_inside (H : ConvexHexagon) (d1 d2 : Diagonal H) : Prop :=
  sorry

/-- The probability of two randomly chosen diagonals intersecting inside the hexagon -/
def intersection_probability (H : ConvexHexagon) : ℚ :=
  sorry

/-- Theorem: The probability of two randomly chosen diagonals intersecting inside a convex hexagon is 5/12 -/
theorem hexagon_diagonal_intersection_probability (H : ConvexHexagon) :
  intersection_probability H = 5 / 12 :=
sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l806_80632


namespace NUMINAMATH_CALUDE_greatest_root_of_f_l806_80671

def f (x : ℝ) := 12 * x^4 - 8 * x^2 + 1

theorem greatest_root_of_f :
  ∃ (r : ℝ), r = Real.sqrt 2 / 2 ∧ 
  f r = 0 ∧ 
  ∀ (x : ℝ), f x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_f_l806_80671


namespace NUMINAMATH_CALUDE_ice_cream_problem_l806_80678

/-- Ice cream purchase and profit maximization problem -/
theorem ice_cream_problem 
  (cost_equation1 : ℝ → ℝ → Prop) 
  (cost_equation2 : ℝ → ℝ → Prop)
  (total_budget : ℝ)
  (total_ice_creams : ℕ)
  (brand_a_constraint : ℕ → ℕ → Prop)
  (selling_price_a : ℝ)
  (selling_price_b : ℝ) :
  -- Part 1: Purchase prices
  ∃ (price_a price_b : ℝ),
    cost_equation1 price_a price_b ∧
    cost_equation2 price_a price_b ∧
    price_a = 12 ∧
    price_b = 15 ∧
  -- Part 2: Profit maximization
  ∃ (brand_a brand_b : ℕ),
    brand_a + brand_b = total_ice_creams ∧
    brand_a_constraint brand_a brand_b ∧
    price_a * brand_a + price_b * brand_b ≤ total_budget ∧
    brand_a = 20 ∧
    brand_b = 20 ∧
    ∀ (m n : ℕ), 
      m + n = total_ice_creams →
      brand_a_constraint m n →
      price_a * m + price_b * n ≤ total_budget →
      (selling_price_a - price_a) * brand_a + (selling_price_b - price_b) * brand_b ≥
      (selling_price_a - price_a) * m + (selling_price_b - price_b) * n :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_problem_l806_80678


namespace NUMINAMATH_CALUDE_farm_milk_production_l806_80653

theorem farm_milk_production
  (a b c d e : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0)
  (he : e > 0)
  (group_a_production : b = (a * c) * (b / (a * c)))
  (group_b_efficiency : ℝ)
  (hefficiency : group_b_efficiency = 1.2)
  : (group_b_efficiency * b * d * e) / (a * c) = (1.2 * b * d * e) / (a * c) :=
by sorry

end NUMINAMATH_CALUDE_farm_milk_production_l806_80653


namespace NUMINAMATH_CALUDE_percentage_of_green_leaves_l806_80645

/-- Given a collection of leaves with known properties, prove the percentage of green leaves. -/
theorem percentage_of_green_leaves 
  (total_leaves : ℕ) 
  (brown_percentage : ℚ) 
  (yellow_leaves : ℕ) 
  (h1 : total_leaves = 25)
  (h2 : brown_percentage = 1/5)
  (h3 : yellow_leaves = 15) :
  (total_leaves - (brown_percentage * total_leaves).num - yellow_leaves : ℚ) / total_leaves = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_green_leaves_l806_80645


namespace NUMINAMATH_CALUDE_seokgi_money_problem_l806_80682

theorem seokgi_money_problem (initial_money : ℕ) : 
  (initial_money / 2) / 2 = 1250 → initial_money = 5000 := by
  sorry

end NUMINAMATH_CALUDE_seokgi_money_problem_l806_80682


namespace NUMINAMATH_CALUDE_expression_value_l806_80656

theorem expression_value : 
  (1/2) * (Real.log 12 / Real.log 3) - (Real.log 2 / Real.log 3) + 
  (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) + 
  ((-2)^4)^(1/4) + (Real.sqrt 3 - 1)^0 = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l806_80656


namespace NUMINAMATH_CALUDE_astronaut_stay_duration_l806_80602

theorem astronaut_stay_duration (days_per_year : ℕ) (seasons_per_year : ℕ) (seasons_stayed : ℕ) : 
  days_per_year = 250 → 
  seasons_per_year = 5 → 
  seasons_stayed = 3 → 
  (days_per_year / seasons_per_year) * seasons_stayed = 150 :=
by sorry

end NUMINAMATH_CALUDE_astronaut_stay_duration_l806_80602


namespace NUMINAMATH_CALUDE_maxwells_speed_l806_80613

/-- Proves that Maxwell's walking speed is 4 km/h given the problem conditions -/
theorem maxwells_speed (total_distance : ℝ) (brad_speed : ℝ) (maxwell_time : ℝ) (brad_time : ℝ) 
  (h1 : total_distance = 14)
  (h2 : brad_speed = 6)
  (h3 : maxwell_time = 2)
  (h4 : brad_time = 1)
  (h5 : maxwell_time * maxwell_speed + brad_time * brad_speed = total_distance) :
  maxwell_speed = 4 := by
  sorry

#check maxwells_speed

end NUMINAMATH_CALUDE_maxwells_speed_l806_80613


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l806_80619

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a line in 3D space
structure Line3D where
  point1 : Point3D
  point2 : Point3D

-- Define the property of non-coplanar points
def nonCoplanar (E F G H : Point3D) : Prop := sorry

-- Define the property of non-intersecting lines
def nonIntersecting (l1 l2 : Line3D) : Prop := sorry

theorem sufficient_but_not_necessary 
  (E F G H : Point3D) 
  (EF : Line3D) 
  (GH : Line3D) 
  (h_EF : EF.point1 = E ∧ EF.point2 = F) 
  (h_GH : GH.point1 = G ∧ GH.point2 = H) :
  (nonCoplanar E F G H → nonIntersecting EF GH) ∧ 
  ∃ E' F' G' H' : Point3D, ∃ EF' GH' : Line3D, 
    (EF'.point1 = E' ∧ EF'.point2 = F') ∧ 
    (GH'.point1 = G' ∧ GH'.point2 = H') ∧ 
    nonIntersecting EF' GH' ∧ 
    ¬(nonCoplanar E' F' G' H') := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l806_80619


namespace NUMINAMATH_CALUDE_mixture_concentration_l806_80672

/-- Represents a vessel with spirit -/
structure Vessel where
  concentration : Rat
  volume : Rat

/-- Calculates the concentration of spirit in a mixture of vessels -/
def mixConcentration (vessels : List Vessel) : Rat :=
  let totalSpirit := vessels.map (λ v => v.concentration * v.volume) |>.sum
  let totalVolume := vessels.map (λ v => v.volume) |>.sum
  totalSpirit / totalVolume

/-- The main theorem stating that the mixture of given vessels results in 26% concentration -/
theorem mixture_concentration : 
  let vessels := [
    Vessel.mk (45/100) 4,
    Vessel.mk (30/100) 5,
    Vessel.mk (10/100) 6
  ]
  mixConcentration vessels = 26/100 := by
  sorry


end NUMINAMATH_CALUDE_mixture_concentration_l806_80672


namespace NUMINAMATH_CALUDE_special_polygon_has_eight_sides_l806_80665

/-- A polygon with n sides where the sum of interior angles is 3 times the sum of exterior angles -/
structure SpecialPolygon where
  n : ℕ
  interior_sum : ℝ
  exterior_sum : ℝ
  h1 : interior_sum = (n - 2) * 180
  h2 : exterior_sum = 360
  h3 : interior_sum = 3 * exterior_sum

/-- Theorem: A SpecialPolygon has 8 sides -/
theorem special_polygon_has_eight_sides (p : SpecialPolygon) : p.n = 8 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_has_eight_sides_l806_80665


namespace NUMINAMATH_CALUDE_specialIntegers_infinite_l806_80620

/-- A function that converts a natural number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- A predicate that checks if a list of digits contains only 1 and 2 -/
def containsOnly1And2 (digits : List ℕ) : Prop :=
  sorry

/-- The set of positive integers n such that n^2 in base 4 contains only digits 1 and 2 -/
def specialIntegers : Set ℕ :=
  {n : ℕ | n > 0 ∧ containsOnly1And2 (toBase4 (n^2))}

/-- The main theorem stating that the set of special integers is infinite -/
theorem specialIntegers_infinite : Set.Infinite specialIntegers :=
  sorry

end NUMINAMATH_CALUDE_specialIntegers_infinite_l806_80620


namespace NUMINAMATH_CALUDE_average_speed_calculation_l806_80663

def total_distance : ℝ := 120
def total_time : ℝ := 7

theorem average_speed_calculation : 
  (total_distance / total_time) = 120 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l806_80663


namespace NUMINAMATH_CALUDE_square_area_problem_l806_80633

theorem square_area_problem (x : ℝ) : 
  (5 * x - 18 = 27 - 4 * x) →
  (5 * x - 18)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_problem_l806_80633


namespace NUMINAMATH_CALUDE_paper_strip_dimensions_l806_80676

theorem paper_strip_dimensions (a b c : ℕ+) (h : 2 * a * b + 2 * a * c - a * a = 43) :
  a = 1 ∧ b + c = 22 := by
  sorry

end NUMINAMATH_CALUDE_paper_strip_dimensions_l806_80676


namespace NUMINAMATH_CALUDE_angle_range_in_triangle_l806_80616

open Real

theorem angle_range_in_triangle (A : ℝ) (h1 : sin A + cos A > 0) (h2 : tan A < sin A) :
  π / 2 < A ∧ A < 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_range_in_triangle_l806_80616


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l806_80698

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem twentieth_term_of_sequence (a₁ a₁₃ a₂₀ : ℝ) :
  a₁ = 3 →
  a₁₃ = 27 →
  (∃ d : ℝ, ∀ n : ℕ, arithmetic_sequence a₁ d n = a₁ + (n - 1 : ℝ) * d) →
  a₂₀ = arithmetic_sequence a₁ ((a₁₃ - a₁) / 12) 20 →
  a₂₀ = 41 := by
sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l806_80698


namespace NUMINAMATH_CALUDE_wooden_block_surface_area_l806_80617

theorem wooden_block_surface_area (A₁ A₂ A₃ A₄ A₅ A₆ A₇ : ℕ) 
  (h₁ : A₁ = 148)
  (h₂ : A₂ = 46)
  (h₃ : A₃ = 72)
  (h₄ : A₄ = 28)
  (h₅ : A₅ = 88)
  (h₆ : A₆ = 126)
  (h₇ : A₇ = 58) :
  ∃ A₈ : ℕ, A₈ = 22 ∧ A₁ + A₂ + A₃ + A₄ - (A₅ + A₆ + A₇) = A₈ :=
by sorry

end NUMINAMATH_CALUDE_wooden_block_surface_area_l806_80617


namespace NUMINAMATH_CALUDE_total_weekly_airflow_l806_80621

/-- Calculates the total airflow generated by three fans in one week -/
theorem total_weekly_airflow (fan_a_flow : ℝ) (fan_a_time : ℝ) 
                              (fan_b_flow : ℝ) (fan_b_time : ℝ) 
                              (fan_c_flow : ℝ) (fan_c_time : ℝ) : 
  fan_a_flow = 10 →
  fan_a_time = 10 →
  fan_b_flow = 15 →
  fan_b_time = 20 →
  fan_c_flow = 25 →
  fan_c_time = 30 →
  ((fan_a_flow * fan_a_time * 60) + 
   (fan_b_flow * fan_b_time * 60) + 
   (fan_c_flow * fan_c_time * 60)) * 7 = 483000 := by
  sorry

#check total_weekly_airflow

end NUMINAMATH_CALUDE_total_weekly_airflow_l806_80621


namespace NUMINAMATH_CALUDE_log_8_problem_l806_80668

theorem log_8_problem (x : ℝ) :
  Real.log x / Real.log 8 = 3.5 → x = 1024 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_8_problem_l806_80668


namespace NUMINAMATH_CALUDE_worker_problem_l806_80683

theorem worker_problem (time_B time_together : ℝ) 
  (h1 : time_B = 10)
  (h2 : time_together = 4.444444444444445)
  (h3 : 1 / time_A + 1 / time_B = 1 / time_together) :
  time_A = 8 :=
sorry

end NUMINAMATH_CALUDE_worker_problem_l806_80683


namespace NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l806_80606

/-- A rectangular solid with dimensions in arithmetic progression -/
structure RectangularSolid where
  a : ℝ  -- middle dimension
  d : ℝ  -- common difference

/-- Volume of the rectangular solid -/
def volume (solid : RectangularSolid) : ℝ :=
  (solid.a - solid.d) * solid.a * (solid.a + solid.d)

/-- Surface area of the rectangular solid -/
def surface_area (solid : RectangularSolid) : ℝ :=
  2 * ((solid.a - solid.d) * solid.a + solid.a * (solid.a + solid.d) + (solid.a - solid.d) * (solid.a + solid.d))

/-- Sum of the lengths of all edges of the rectangular solid -/
def sum_of_edges (solid : RectangularSolid) : ℝ :=
  4 * ((solid.a - solid.d) + solid.a + (solid.a + solid.d))

theorem rectangular_solid_edge_sum
  (solid : RectangularSolid)
  (h_volume : volume solid = 512)
  (h_area : surface_area solid = 352) :
  sum_of_edges solid = 12 * Real.sqrt 59 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l806_80606


namespace NUMINAMATH_CALUDE_orange_cost_calculation_l806_80618

def initial_amount : ℕ := 95
def apple_cost : ℕ := 25
def candy_cost : ℕ := 6
def amount_left : ℕ := 50

theorem orange_cost_calculation : 
  initial_amount - amount_left - apple_cost - candy_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_calculation_l806_80618


namespace NUMINAMATH_CALUDE_second_number_is_fifteen_l806_80628

def has_exactly_three_common_factors_with_15 (n : ℕ) : Prop :=
  ∃ (f₁ f₂ f₃ : ℕ), 
    f₁ ≠ f₂ ∧ f₁ ≠ f₃ ∧ f₂ ≠ f₃ ∧
    f₁ > 1 ∧ f₂ > 1 ∧ f₃ > 1 ∧
    f₁ ∣ 15 ∧ f₂ ∣ 15 ∧ f₃ ∣ 15 ∧
    f₁ ∣ n ∧ f₂ ∣ n ∧ f₃ ∣ n ∧
    ∀ (k : ℕ), k > 1 → k ∣ 15 → k ∣ n → (k = f₁ ∨ k = f₂ ∨ k = f₃)

theorem second_number_is_fifteen (n : ℕ) 
  (h : has_exactly_three_common_factors_with_15 n)
  (h3 : 3 ∣ n) (h5 : 5 ∣ n) (h15 : 15 ∣ n) : n = 15 :=
by sorry

end NUMINAMATH_CALUDE_second_number_is_fifteen_l806_80628


namespace NUMINAMATH_CALUDE_paper_products_distribution_l806_80667

theorem paper_products_distribution (total : ℕ) 
  (h1 : total = 20)
  (h2 : total / 2 + total / 4 + total / 5 + paper_cups = total) : 
  paper_cups = 1 := by
  sorry

end NUMINAMATH_CALUDE_paper_products_distribution_l806_80667


namespace NUMINAMATH_CALUDE_f_of_g_3_l806_80607

def g (x : ℝ) : ℝ := 4 * x + 5

def f (x : ℝ) : ℝ := 6 * x - 11

theorem f_of_g_3 : f (g 3) = 91 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_3_l806_80607


namespace NUMINAMATH_CALUDE_minimum_g_5_l806_80644

def Tenuous (f : ℕ → ℤ) : Prop :=
  ∀ x y : ℕ, x > 0 ∧ y > 0 → f x + f y > y^2

def SumOfG (g : ℕ → ℤ) : ℤ :=
  (List.range 10).map (λ i => g (i + 1)) |>.sum

theorem minimum_g_5 (g : ℕ → ℤ) (h_tenuous : Tenuous g) 
    (h_min : ∀ g' : ℕ → ℤ, Tenuous g' → SumOfG g ≤ SumOfG g') : 
  g 5 ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_minimum_g_5_l806_80644


namespace NUMINAMATH_CALUDE_sqrt_equation_implies_difference_l806_80692

theorem sqrt_equation_implies_difference (m n : ℕ) : 
  (Real.sqrt (9 - m / n) = 9 * Real.sqrt (m / n)) → (n - m = 73) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_implies_difference_l806_80692


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l806_80612

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ (s : Finset ℤ), s.card = 5 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x : ℝ) - a ≤ 0 ∧ 7 + 2 * (x : ℝ) > 1)) →
  2 ≤ a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l806_80612


namespace NUMINAMATH_CALUDE_lollipop_distribution_l806_80655

theorem lollipop_distribution (raspberry mint orange cotton_candy : ℕ) 
  (h1 : raspberry = 60) 
  (h2 : mint = 135) 
  (h3 : orange = 5) 
  (h4 : cotton_candy = 330) 
  (friends : ℕ) 
  (h5 : friends = 15) : 
  (raspberry + mint + orange + cotton_candy) % friends = 5 := by
sorry

end NUMINAMATH_CALUDE_lollipop_distribution_l806_80655


namespace NUMINAMATH_CALUDE_oncoming_train_speed_l806_80662

/-- Given two trains passing each other, calculate the speed of the oncoming train -/
theorem oncoming_train_speed
  (v₁ : ℝ)  -- Speed of the passenger's train in km/h
  (l : ℝ)   -- Length of the oncoming train in meters
  (t : ℝ)   -- Time taken for the oncoming train to pass in seconds
  (h₁ : v₁ = 40)  -- The speed of the passenger's train is 40 km/h
  (h₂ : l = 75)   -- The length of the oncoming train is 75 meters
  (h₃ : t = 3)    -- The time taken to pass is 3 seconds
  : ∃ v₂ : ℝ, v₂ = 50 ∧ l / 1000 = (v₁ + v₂) * (t / 3600) :=
sorry

end NUMINAMATH_CALUDE_oncoming_train_speed_l806_80662


namespace NUMINAMATH_CALUDE_triangle_area_l806_80642

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  b = 6 →
  a = 2 * c →
  B = π / 3 →
  (1 / 2) * a * c * Real.sin B = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l806_80642


namespace NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l806_80695

theorem cost_to_selling_price_ratio (cost_price selling_price : ℝ) 
  (h_profit : selling_price = cost_price * (1 + 0.25)) :
  cost_price / selling_price = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l806_80695


namespace NUMINAMATH_CALUDE_lemonade_water_quarts_l806_80685

/-- Proves the number of quarts of water needed for a special lemonade recipe -/
theorem lemonade_water_quarts : 
  let total_parts : ℚ := 5 + 3
  let water_parts : ℚ := 5
  let total_gallons : ℚ := 5
  let quarts_per_gallon : ℚ := 4
  (water_parts / total_parts) * total_gallons * quarts_per_gallon = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_water_quarts_l806_80685


namespace NUMINAMATH_CALUDE_train_crossing_time_l806_80661

/-- The time taken for a train to cross a pole -/
theorem train_crossing_time (train_speed_kmh : ℝ) (train_length_m : ℝ) : 
  train_speed_kmh = 72 → train_length_m = 180 → 
  (train_length_m / (train_speed_kmh * 1000 / 3600)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l806_80661


namespace NUMINAMATH_CALUDE_square_value_l806_80657

theorem square_value (p : ℝ) (h1 : p + p = 75) (h2 : (p + p) + 2*p = 149) : p = 38 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l806_80657


namespace NUMINAMATH_CALUDE_camp_gender_ratio_l806_80694

theorem camp_gender_ratio (total : ℕ) (boys_added : ℕ) (girls_percent : ℝ) : 
  total = 100 → 
  boys_added = 100 → 
  girls_percent = 5 → 
  (total : ℝ) * girls_percent / 100 = (total - ((total + boys_added) * girls_percent / 100)) → 
  (100 : ℝ) * (total - ((total + boys_added) * girls_percent / 100)) / total = 90 :=
by sorry

end NUMINAMATH_CALUDE_camp_gender_ratio_l806_80694


namespace NUMINAMATH_CALUDE_jane_morning_reading_l806_80614

/-- The number of pages Jane reads in the morning -/
def morning_pages : ℕ := 5

/-- The number of pages Jane reads in the evening -/
def evening_pages : ℕ := 10

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pages Jane reads in a week -/
def total_pages : ℕ := 105

/-- Theorem stating that Jane reads 5 pages in the morning -/
theorem jane_morning_reading :
  morning_pages = 5 ∧
  evening_pages = 10 ∧
  days_in_week = 7 ∧
  total_pages = 105 ∧
  days_in_week * (morning_pages + evening_pages) = total_pages :=
by sorry

end NUMINAMATH_CALUDE_jane_morning_reading_l806_80614


namespace NUMINAMATH_CALUDE_fish_in_third_tank_l806_80615

/-- The number of fish in the first tank -/
def first_tank : ℕ := 7 + 8

/-- The number of fish in the second tank -/
def second_tank : ℕ := 2 * first_tank

/-- The number of fish in the third tank -/
def third_tank : ℕ := second_tank / 3

theorem fish_in_third_tank : third_tank = 10 := by
  sorry

end NUMINAMATH_CALUDE_fish_in_third_tank_l806_80615


namespace NUMINAMATH_CALUDE_quadratic_polynomial_remainders_l806_80674

theorem quadratic_polynomial_remainders (m n : ℚ) : 
  (∀ x, (x^2 + m*x + n) % (x - m) = m ∧ (x^2 + m*x + n) % (x - n) = n) ↔ 
  ((m = 0 ∧ n = 0) ∨ (m = 1/2 ∧ n = 0) ∨ (m = 1 ∧ n = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_remainders_l806_80674


namespace NUMINAMATH_CALUDE_blue_balls_count_l806_80646

theorem blue_balls_count (total : ℕ) (prob : ℚ) (blue : ℕ) : 
  total = 15 →
  prob = 1 / 21 →
  (blue * (blue - 1)) / (total * (total - 1)) = prob →
  blue = 5 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l806_80646


namespace NUMINAMATH_CALUDE_heartsuit_properties_l806_80651

/-- The heartsuit operation on real numbers -/
def heartsuit (x y : ℝ) : ℝ := 2 * |x - y|

/-- Properties of the heartsuit operation -/
theorem heartsuit_properties :
  (∀ x y : ℝ, heartsuit x y = heartsuit y x) ∧ 
  (∀ x y : ℝ, 3 * (heartsuit x y) = heartsuit (3*x) (3*y)) ∧ 
  (∀ x : ℝ, heartsuit x x = 0) ∧ 
  (∀ x y : ℝ, x ≠ y → heartsuit x y > 0) ∧ 
  (∀ x : ℝ, x ≥ 0 → heartsuit x 0 = 2*x) ∧
  (∃ x : ℝ, heartsuit x 0 ≠ 2*x) :=
by sorry

end NUMINAMATH_CALUDE_heartsuit_properties_l806_80651


namespace NUMINAMATH_CALUDE_cube_difference_identity_l806_80652

theorem cube_difference_identity (a b : ℝ) : 
  (a^3 + b^3 = (a + b) * (a^2 - a*b + b^2)) → 
  (a^3 - b^3 = (a - b) * (a^2 + a*b + b^2)) := by
sorry

end NUMINAMATH_CALUDE_cube_difference_identity_l806_80652


namespace NUMINAMATH_CALUDE_inequality_solution_set_l806_80648

-- Define the solution set
def solution_set := {x : ℝ | x < 5}

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | |x - 8| - |x - 4| > 2} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l806_80648


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l806_80696

theorem fly_distance_from_ceiling (z : ℝ) : 
  (2:ℝ)^2 + 6^2 + z^2 = 11^2 → z = 9 := by
  sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l806_80696


namespace NUMINAMATH_CALUDE_closest_multiple_of_15_to_1987_l806_80611

theorem closest_multiple_of_15_to_1987 :
  ∃ (n : ℤ), n * 15 = 1980 ∧
  ∀ (m : ℤ), m * 15 ≠ 1980 → |1987 - (m * 15)| ≥ |1987 - 1980| := by
  sorry

end NUMINAMATH_CALUDE_closest_multiple_of_15_to_1987_l806_80611


namespace NUMINAMATH_CALUDE_largest_valid_n_l806_80669

def engineers : Nat := 6
def technicians : Nat := 12
def workers : Nat := 18

def total_individuals : Nat := engineers + technicians + workers

def is_valid_n (n : Nat) : Prop :=
  n ∣ total_individuals ∧
  n ≤ Nat.lcm (Nat.lcm engineers technicians) workers ∧
  ¬((n + 1) ∣ total_individuals)

theorem largest_valid_n :
  ∃ (n : Nat), is_valid_n n ∧ ∀ (m : Nat), is_valid_n m → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_valid_n_l806_80669


namespace NUMINAMATH_CALUDE_third_month_sale_l806_80670

def average_sale : ℕ := 7500
def num_months : ℕ := 6
def sale_month1 : ℕ := 7435
def sale_month2 : ℕ := 7927
def sale_month4 : ℕ := 8230
def sale_month5 : ℕ := 7562
def sale_month6 : ℕ := 5991

theorem third_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6
  total_sales - known_sales = 7855 := by
sorry

end NUMINAMATH_CALUDE_third_month_sale_l806_80670


namespace NUMINAMATH_CALUDE_pencil_count_l806_80640

theorem pencil_count (num_boxes : ℝ) (pencils_per_box : ℝ) (h1 : num_boxes = 4.0) (h2 : pencils_per_box = 648.0) :
  num_boxes * pencils_per_box = 2592.0 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l806_80640


namespace NUMINAMATH_CALUDE_shaded_area_is_one_third_l806_80664

/-- Two rectangles with dimensions 10 × 20 overlap to form a 20 × 30 rectangle. -/
structure OverlappingRectangles where
  small_width : ℝ
  small_height : ℝ
  large_width : ℝ
  large_height : ℝ
  small_width_eq : small_width = 10
  small_height_eq : small_height = 20
  large_width_eq : large_width = 20
  large_height_eq : large_height = 30

/-- The shaded area is the overlap of the two smaller rectangles. -/
def shaded_area (r : OverlappingRectangles) : ℝ :=
  r.small_width * r.small_height

/-- The area of the larger rectangle. -/
def large_area (r : OverlappingRectangles) : ℝ :=
  r.large_width * r.large_height

/-- The theorem stating that the shaded area is 1/3 of the larger rectangle's area. -/
theorem shaded_area_is_one_third (r : OverlappingRectangles) :
    shaded_area r / large_area r = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_one_third_l806_80664


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l806_80660

-- Define the set of cards
inductive Card : Type
| Hearts : Card
| Spades : Card
| Diamonds : Card
| Clubs : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the event "A receives a club"
def A_receives_club (d : Distribution) : Prop := d Person.A = Card.Clubs

-- Define the event "B receives a club"
def B_receives_club (d : Distribution) : Prop := d Person.B = Card.Clubs

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(A_receives_club d ∧ B_receives_club d)) ∧
  (∃ d : Distribution, ¬(A_receives_club d ∨ B_receives_club d)) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l806_80660


namespace NUMINAMATH_CALUDE_lauras_workout_speed_l806_80686

theorem lauras_workout_speed :
  ∃! x : ℝ, x > 0 ∧ (25 / (3 * x + 2)) + (8 / x) = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_lauras_workout_speed_l806_80686


namespace NUMINAMATH_CALUDE_house_sale_profit_l806_80680

theorem house_sale_profit (market_value : ℝ) (over_market_percentage : ℝ) 
  (tax_rate : ℝ) (num_people : ℕ) : 
  market_value = 500000 ∧ 
  over_market_percentage = 0.2 ∧ 
  tax_rate = 0.1 ∧ 
  num_people = 4 → 
  (market_value * (1 + over_market_percentage) * (1 - tax_rate)) / num_people = 135000 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_profit_l806_80680


namespace NUMINAMATH_CALUDE_fenced_area_calculation_l806_80681

theorem fenced_area_calculation : 
  let yard_length : ℕ := 20
  let yard_width : ℕ := 18
  let large_cutout_side : ℕ := 4
  let small_cutout_side : ℕ := 2
  let yard_area := yard_length * yard_width
  let large_cutout_area := large_cutout_side * large_cutout_side
  let small_cutout_area := small_cutout_side * small_cutout_side
  yard_area - large_cutout_area - small_cutout_area = 340 := by
sorry

end NUMINAMATH_CALUDE_fenced_area_calculation_l806_80681


namespace NUMINAMATH_CALUDE_nursing_home_milk_distribution_l806_80608

theorem nursing_home_milk_distribution (elderly : ℕ) (milk : ℕ) : 
  (2 * elderly + 16 = milk) ∧ (4 * elderly = milk + 12) → 
  (elderly = 14 ∧ milk = 44) :=
by sorry

end NUMINAMATH_CALUDE_nursing_home_milk_distribution_l806_80608


namespace NUMINAMATH_CALUDE_digit_puzzle_solution_l806_80641

/-- Represents a digit in base 10 -/
def Digit := Fin 10

/-- Checks if all elements in a list are distinct -/
def all_distinct (l : List Digit) : Prop :=
  ∀ i j, i ≠ j → l.get i ≠ l.get j

/-- Converts a pair of digits to a two-digit number -/
def to_number (tens digit : Digit) : Nat :=
  10 * tens.val + digit.val

/-- The main theorem -/
theorem digit_puzzle_solution (Y E M T : Digit) 
  (h_distinct : all_distinct [Y, E, M, T])
  (h_equation : to_number Y E * to_number M E = to_number T T * 101) :
  E.val + M.val + T.val + Y.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_puzzle_solution_l806_80641


namespace NUMINAMATH_CALUDE_cricket_team_captain_age_l806_80610

theorem cricket_team_captain_age (team_size : ℕ) (whole_team_avg_age : ℕ) 
  (captain_age wicket_keeper_age : ℕ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 3 →
  whole_team_avg_age = 21 →
  (whole_team_avg_age * team_size - captain_age - wicket_keeper_age) / (team_size - 2) + 1 = whole_team_avg_age →
  captain_age = 24 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_captain_age_l806_80610


namespace NUMINAMATH_CALUDE_subsets_containing_five_and_seven_l806_80697

def S : Finset Nat := {1, 2, 3, 4, 5, 6, 7}

theorem subsets_containing_five_and_seven :
  (Finset.filter (fun s => 5 ∈ s ∧ 7 ∈ s) (Finset.powerset S)).card = 32 := by
  sorry

end NUMINAMATH_CALUDE_subsets_containing_five_and_seven_l806_80697


namespace NUMINAMATH_CALUDE_exponent_calculation_l806_80624

theorem exponent_calculation : (3^5 / 3^2) * 5^6 = 421875 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l806_80624


namespace NUMINAMATH_CALUDE_all_transformed_in_R_l806_80605

def R : Set ℂ := {z | -2 ≤ z.re ∧ z.re ≤ 2 ∧ -1 ≤ z.im ∧ z.im ≤ 1}

theorem all_transformed_in_R : ∀ z ∈ R, (1/2 + 1/2*I) * z ∈ R := by
  sorry

end NUMINAMATH_CALUDE_all_transformed_in_R_l806_80605


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l806_80622

/-- Given a hyperbola E with the standard equation x²/4 - y² = 1,
    prove that the equations of its asymptotes are y = ± (1/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 4 - y^2 = 1) →
  (∃ (k : ℝ), k = 1/2 ∧ (y = k*x ∨ y = -k*x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l806_80622


namespace NUMINAMATH_CALUDE_lattice_triangle_area_l806_80639

/-- A lattice point is a point with integer coordinates. -/
def LatticePoint (p : ℝ × ℝ) : Prop := ∃ (x y : ℤ), p = (↑x, ↑y)

/-- A triangle with vertices at lattice points. -/
structure LatticeTriangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v1_lattice : LatticePoint v1
  v2_lattice : LatticePoint v2
  v3_lattice : LatticePoint v3

/-- The number of lattice points strictly inside the triangle. -/
def interior_points (t : LatticeTriangle) : ℕ := sorry

/-- The number of lattice points on the sides of the triangle (excluding vertices). -/
def boundary_points (t : LatticeTriangle) : ℕ := sorry

/-- The area of a triangle. -/
def triangle_area (t : LatticeTriangle) : ℝ := sorry

/-- Theorem: The area of a lattice triangle with n interior points and m boundary points
    (excluding vertices) is equal to n + m/2 + 1/2. -/
theorem lattice_triangle_area (t : LatticeTriangle) :
  triangle_area t = interior_points t + (boundary_points t : ℝ) / 2 + 1 / 2 := by sorry

end NUMINAMATH_CALUDE_lattice_triangle_area_l806_80639


namespace NUMINAMATH_CALUDE_ian_lottery_payment_l806_80601

theorem ian_lottery_payment (total : ℝ) (left : ℝ) (colin : ℝ) (helen : ℝ) (benedict : ℝ) :
  total = 100 →
  helen = 2 * colin →
  benedict = helen / 2 →
  left = 20 →
  total = colin + helen + benedict + left →
  colin = 20 := by
sorry

end NUMINAMATH_CALUDE_ian_lottery_payment_l806_80601


namespace NUMINAMATH_CALUDE_m_intersect_n_equals_open_one_closed_three_l806_80693

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 9}

-- State the theorem
theorem m_intersect_n_equals_open_one_closed_three : M ∩ N = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_m_intersect_n_equals_open_one_closed_three_l806_80693


namespace NUMINAMATH_CALUDE_division_problem_l806_80677

theorem division_problem (N D Q R : ℕ) : 
  D = 5 → Q = 4 → R = 3 → N = D * Q + R → N = 23 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l806_80677


namespace NUMINAMATH_CALUDE_function_equivalence_l806_80691

-- Define the function f
noncomputable def f : ℝ → ℝ :=
  fun x => -x^2 + 1/x - 2

-- State the theorem
theorem function_equivalence (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 1) (hx3 : x ≠ -1) :
  f (x - 1/x) = x / (x^2 - 1) - x^2 - 1/x^2 :=
by
  sorry


end NUMINAMATH_CALUDE_function_equivalence_l806_80691


namespace NUMINAMATH_CALUDE_sodium_carbonate_mass_fraction_l806_80654

/-- Given the number of moles of Na₂CO₃, its molar mass, and the total mass of the solution,
    prove that the mass fraction of Na₂CO₃ in the solution is 10%. -/
theorem sodium_carbonate_mass_fraction 
  (n : Real) 
  (M : Real) 
  (m_solution : Real) 
  (h1 : n = 0.125) 
  (h2 : M = 106) 
  (h3 : m_solution = 132.5) : 
  (n * M * 100 / m_solution) = 10 := by
  sorry

#check sodium_carbonate_mass_fraction

end NUMINAMATH_CALUDE_sodium_carbonate_mass_fraction_l806_80654


namespace NUMINAMATH_CALUDE_school_ratio_l806_80699

-- Define the school structure
structure School where
  b : ℕ  -- number of teachers
  c : ℕ  -- number of students
  k : ℕ  -- number of students each teacher teaches
  h : ℕ  -- number of teachers teaching any two different students

-- Define the theorem
theorem school_ratio (s : School) : 
  s.b / s.h = (s.c * (s.c - 1)) / (s.k * (s.k - 1)) := by
  sorry

end NUMINAMATH_CALUDE_school_ratio_l806_80699


namespace NUMINAMATH_CALUDE_jug_emptying_l806_80658

theorem jug_emptying (Cx Cy Cz : ℝ) (hx : Cx > 0) (hy : Cy > 0) (hz : Cz > 0) :
  let initial_x := (1/4 : ℝ) * Cx
  let initial_y := (2/3 : ℝ) * Cy
  let initial_z := (3/5 : ℝ) * Cz
  let water_to_fill_y := Cy - initial_y
  let remaining_x := initial_x - water_to_fill_y
  remaining_x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_jug_emptying_l806_80658


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l806_80638

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 - x - 2 = 0 → x = 2 ∨ x = -1) ↔
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -1 → x^2 - x - 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l806_80638


namespace NUMINAMATH_CALUDE_square_minus_four_times_plus_four_equals_six_l806_80609

theorem square_minus_four_times_plus_four_equals_six (a : ℝ) :
  a = Real.sqrt 6 + 2 → a^2 - 4*a + 4 = 6 := by sorry

end NUMINAMATH_CALUDE_square_minus_four_times_plus_four_equals_six_l806_80609


namespace NUMINAMATH_CALUDE_dave_tickets_l806_80690

theorem dave_tickets (tickets_used : ℕ) (tickets_left : ℕ) : 
  tickets_used = 6 → tickets_left = 7 → tickets_used + tickets_left = 13 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_l806_80690


namespace NUMINAMATH_CALUDE_fifteen_cells_covered_by_two_l806_80649

/-- Represents a square on a graph paper --/
structure Square :=
  (side : ℕ)

/-- Represents the configuration of squares on the graph paper --/
structure SquareConfiguration :=
  (squares : List Square)
  (total_area : ℕ)
  (unique_area : ℕ)
  (triple_overlap : ℕ)

/-- Calculates the number of cells covered by exactly two squares --/
def cells_covered_by_two (config : SquareConfiguration) : ℕ :=
  config.total_area - config.unique_area - 2 * config.triple_overlap

/-- Theorem stating that for the given configuration, 15 cells are covered by exactly two squares --/
theorem fifteen_cells_covered_by_two (config : SquareConfiguration) 
  (h1 : config.squares.length = 3)
  (h2 : ∀ s ∈ config.squares, s.side = 5)
  (h3 : config.total_area = 75)
  (h4 : config.unique_area = 56)
  (h5 : config.triple_overlap = 2) :
  cells_covered_by_two config = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_cells_covered_by_two_l806_80649


namespace NUMINAMATH_CALUDE_function_is_identity_l806_80666

def IsNonDegenerateTriangle (a b c : ℕ+) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def SatisfiesTriangleCondition (f : ℕ+ → ℕ+) : Prop :=
  ∀ a b : ℕ+, IsNonDegenerateTriangle a (f b) (f (b + f a - 1))

theorem function_is_identity (f : ℕ+ → ℕ+) 
  (h : SatisfiesTriangleCondition f) : 
  ∀ x : ℕ+, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_is_identity_l806_80666


namespace NUMINAMATH_CALUDE_prob_at_least_one_two_l806_80625

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

/-- The number of outcomes where neither die shows a 2 -/
def neitherShowsTwo : ℕ := (numSides - 1) * (numSides - 1)

/-- The number of outcomes where at least one die shows a 2 -/
def atLeastOneShowsTwo : ℕ := totalOutcomes - neitherShowsTwo

/-- The probability of at least one die showing a 2 when two fair 6-sided dice are rolled -/
theorem prob_at_least_one_two : 
  (atLeastOneShowsTwo : ℚ) / totalOutcomes = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_two_l806_80625


namespace NUMINAMATH_CALUDE_annual_interest_calculation_l806_80687

theorem annual_interest_calculation (principal : ℝ) (quarterly_rate : ℝ) :
  principal = 10000 →
  quarterly_rate = 0.05 →
  (principal * quarterly_rate * 4) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_annual_interest_calculation_l806_80687


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_and_c_l806_80684

/-- Given a quadratic equation x^2 - 6x + c = 0 with one root being 2,
    prove that the other root is 4 and the value of c is 8. -/
theorem quadratic_equation_roots_and_c (c : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + c = 0 ∧ x = 2) →
  (∃ y : ℝ, y^2 - 6*y + c = 0 ∧ y = 4 ∧ c = 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_and_c_l806_80684


namespace NUMINAMATH_CALUDE_base_7_23456_equals_6068_l806_80650

def base_7_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_7_23456_equals_6068 :
  base_7_to_10 [6, 5, 4, 3, 2] = 6068 := by
  sorry

end NUMINAMATH_CALUDE_base_7_23456_equals_6068_l806_80650


namespace NUMINAMATH_CALUDE_system_inequality_solution_l806_80636

theorem system_inequality_solution (a : ℝ) : 
  (∃ x : ℝ, x > 2 ∧ x > -1 ∧ x > a) → ∀ b : ℝ, ∃ x : ℝ, x > 2 ∧ x > -1 ∧ x > b :=
by sorry

end NUMINAMATH_CALUDE_system_inequality_solution_l806_80636


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l806_80659

theorem fahrenheit_to_celsius (C F : ℝ) : C = (5/9) * (F - 32) → C = 40 → F = 104 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l806_80659


namespace NUMINAMATH_CALUDE_price_decrease_l806_80679

theorem price_decrease (original_price : ℝ) : 
  (original_price * (1 - 0.24) = 836) → original_price = 1100 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_l806_80679


namespace NUMINAMATH_CALUDE_length_F_to_F_prime_l806_80688

/-- Triangle DEF with vertices D(-1, 3), E(5, -1), and F(-4, -2) is reflected over the y-axis.
    This theorem proves that the length of the segment from F to F' is 8. -/
theorem length_F_to_F_prime (D E F : ℝ × ℝ) : 
  D = (-1, 3) → E = (5, -1) → F = (-4, -2) → 
  let F' := (-(F.1), F.2)
  abs (F'.1 - F.1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_length_F_to_F_prime_l806_80688


namespace NUMINAMATH_CALUDE_m_range_l806_80643

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) ↔ (m ≥ 4 ∨ m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_m_range_l806_80643


namespace NUMINAMATH_CALUDE_quadratic_form_j_value_l806_80600

theorem quadratic_form_j_value 
  (a b c : ℝ) 
  (h1 : ∀ x, a * x^2 + b * x + c = 5 * (x - 3)^2 + 15) 
  (m n j : ℝ) 
  (h2 : ∀ x, 4 * (a * x^2 + b * x + c) = m * (x - j)^2 + n) : 
  j = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_j_value_l806_80600


namespace NUMINAMATH_CALUDE_line_transformation_l806_80637

-- Define the original line
def original_line (x : ℝ) : ℝ := x - 2

-- Define the transformation (moving 3 units upwards)
def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x => f x + 3

-- Define the new line
def new_line (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

-- Theorem statement
theorem line_transformation :
  ∃ k b : ℝ, 
    (∀ x : ℝ, transform original_line x = new_line k b x) ∧ 
    k = 1 ∧ 
    b = 1 ∧ 
    (∀ x : ℝ, new_line k b x > 0 → x > -1) :=
by sorry

end NUMINAMATH_CALUDE_line_transformation_l806_80637


namespace NUMINAMATH_CALUDE_age_problem_l806_80631

/-- Given a group of 7 people, if adding a person of age x increases the average age by 2,
    and adding a person aged 15 decreases the average age by 1, then x = 39. -/
theorem age_problem (T : ℝ) (A : ℝ) (x : ℝ) : 
  T = 7 * A →
  T + x = 8 * (A + 2) →
  T + 15 = 8 * (A - 1) →
  x = 39 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l806_80631
