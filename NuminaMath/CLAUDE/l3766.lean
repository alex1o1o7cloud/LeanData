import Mathlib

namespace NUMINAMATH_CALUDE_num_paths_equals_1960_l3766_376667

/-- The number of paths from A to D passing through C in a 7x9 grid -/
def num_paths_A_to_D_via_C : ℕ :=
  let grid_width := 7
  let grid_height := 9
  let C_right := 4
  let C_down := 3
  let paths_A_to_C := Nat.choose (C_right + C_down) C_right
  let paths_C_to_D := Nat.choose ((grid_width - C_right) + (grid_height - C_down)) (grid_height - C_down)
  paths_A_to_C * paths_C_to_D

/-- Theorem stating that the number of 15-step paths from A to D passing through C is 1960 -/
theorem num_paths_equals_1960 : num_paths_A_to_D_via_C = 1960 := by
  sorry

end NUMINAMATH_CALUDE_num_paths_equals_1960_l3766_376667


namespace NUMINAMATH_CALUDE_cost_price_percentage_l3766_376637

theorem cost_price_percentage (marked_price cost_price selling_price : ℝ) : 
  marked_price > 0 →
  cost_price > 0 →
  selling_price = marked_price * 0.9 →
  selling_price = cost_price * (1 + 20 / 700) →
  cost_price / marked_price = 0.875 := by
sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l3766_376637


namespace NUMINAMATH_CALUDE_cos_sin_inequality_range_l3766_376614

theorem cos_sin_inequality_range (θ : Real) :
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  Real.cos θ ^ 5 - Real.sin θ ^ 5 < 7 * (Real.sin θ ^ 3 - Real.cos θ ^ 3) →
  θ ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_inequality_range_l3766_376614


namespace NUMINAMATH_CALUDE_nested_radical_value_l3766_376620

/-- The value of the infinite nested radical sqrt(3 - sqrt(3 - sqrt(3 - ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt 3))))))

/-- Theorem stating that the nested radical equals (-1 + √13) / 2 -/
theorem nested_radical_value : nestedRadical = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l3766_376620


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3766_376625

theorem arithmetic_geometric_mean_inequality 
  (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a + b > 2 * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3766_376625


namespace NUMINAMATH_CALUDE_runner_stop_time_l3766_376657

theorem runner_stop_time (total_distance : ℝ) (first_pace second_pace stop_time : ℝ) 
  (h1 : total_distance = 10)
  (h2 : first_pace = 8)
  (h3 : second_pace = 7)
  (h4 : stop_time = 8)
  (h5 : first_pace > second_pace)
  (h6 : stop_time / (first_pace - second_pace) + 
        (stop_time / (first_pace - second_pace)) * second_pace = total_distance) :
  (stop_time / (first_pace - second_pace)) * second_pace = 56 := by
  sorry


end NUMINAMATH_CALUDE_runner_stop_time_l3766_376657


namespace NUMINAMATH_CALUDE_boys_count_l3766_376658

/-- Represents the number of skips in a jump rope competition. -/
structure SkipCompetition where
  boyAvg : ℕ
  girlAvg : ℕ
  totalAvg : ℕ
  boyCount : ℕ
  girlCount : ℕ

/-- Theorem stating the number of boys in the skip competition. -/
theorem boys_count (comp : SkipCompetition) 
  (h1 : comp.boyAvg = 85)
  (h2 : comp.girlAvg = 92)
  (h3 : comp.totalAvg = 88)
  (h4 : comp.boyCount = comp.girlCount + 10)
  (h5 : (comp.boyAvg * comp.boyCount + comp.girlAvg * comp.girlCount) / (comp.boyCount + comp.girlCount) = comp.totalAvg) :
  comp.boyCount = 40 := by
  sorry

end NUMINAMATH_CALUDE_boys_count_l3766_376658


namespace NUMINAMATH_CALUDE_max_k_value_l3766_376690

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, (1/m + 2/(1-2*m) ≥ k) → k ≤ 8) ∧ 
  ∃ k : ℝ, k = 8 ∧ 1/m + 2/(1-2*m) ≥ k :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3766_376690


namespace NUMINAMATH_CALUDE_expression_simplification_l3766_376681

theorem expression_simplification (a b x : ℝ) :
  (Real.sqrt (a^2 + b^2 + x^2) - (x^2 - b^2 - a^2) / Real.sqrt (a^2 + b^2 + x^2)) / (a^2 + b^2 + x^2) =
  2 * (a^2 + b^2) / (a^2 + b^2 + x^2)^(3/2) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3766_376681


namespace NUMINAMATH_CALUDE_cut_into_three_similar_rectangles_l3766_376693

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Checks if two rectangles are similar -/
def similar (r1 r2 : Rectangle) : Prop :=
  r1.width / r1.height = r2.width / r2.height

/-- The original rectangle -/
def originalRect : Rectangle :=
  { width := 10, height := 9 }

/-- Theorem stating that the original rectangle can be cut into three unequal but similar rectangles -/
theorem cut_into_three_similar_rectangles :
  ∃ (r1 r2 r3 : Rectangle),
    r1.width + r2.width + r3.width = originalRect.width ∧
    r1.height + r2.height + r3.height = originalRect.height ∧
    r1.width ≠ r2.width ∧ r2.width ≠ r3.width ∧ r1.width ≠ r3.width ∧
    similar r1 r2 ∧ similar r2 r3 ∧ similar r1 r3 ∧
    similar r1 originalRect ∧ similar r2 originalRect ∧ similar r3 originalRect :=
  by sorry

end NUMINAMATH_CALUDE_cut_into_three_similar_rectangles_l3766_376693


namespace NUMINAMATH_CALUDE_candidates_calculation_l3766_376652

theorem candidates_calculation (candidates : ℕ) : 
  (candidates * 7 / 100 = candidates * 6 / 100 + 83) → 
  candidates = 8300 := by
sorry

end NUMINAMATH_CALUDE_candidates_calculation_l3766_376652


namespace NUMINAMATH_CALUDE_total_revenue_is_99_l3766_376646

def cookies_baked : ℕ := 60
def brownies_baked : ℕ := 32
def cookies_eaten_kyle : ℕ := 2
def brownies_eaten_kyle : ℕ := 2
def cookies_eaten_mom : ℕ := 1
def brownies_eaten_mom : ℕ := 2
def cookie_price : ℚ := 1
def brownie_price : ℚ := 3/2

theorem total_revenue_is_99 :
  let cookies_left := cookies_baked - (cookies_eaten_kyle + cookies_eaten_mom)
  let brownies_left := brownies_baked - (brownies_eaten_kyle + brownies_eaten_mom)
  let revenue_cookies := (cookies_left : ℚ) * cookie_price
  let revenue_brownies := (brownies_left : ℚ) * brownie_price
  revenue_cookies + revenue_brownies = 99
  := by sorry

end NUMINAMATH_CALUDE_total_revenue_is_99_l3766_376646


namespace NUMINAMATH_CALUDE_company_production_l3766_376696

/-- Calculates the total number of bottles produced in one day given the number of cases and bottles per case. -/
def bottles_per_day (cases : ℕ) (bottles_per_case : ℕ) : ℕ :=
  cases * bottles_per_case

/-- Theorem stating that given the specific conditions, the company produces 72,000 bottles per day. -/
theorem company_production : bottles_per_day 7200 10 = 72000 := by
  sorry

end NUMINAMATH_CALUDE_company_production_l3766_376696


namespace NUMINAMATH_CALUDE_cards_per_layer_calculation_l3766_376663

def number_of_decks : ℕ := 16
def cards_per_deck : ℕ := 52
def number_of_layers : ℕ := 32

def total_cards : ℕ := number_of_decks * cards_per_deck

theorem cards_per_layer_calculation :
  total_cards / number_of_layers = 26 := by sorry

end NUMINAMATH_CALUDE_cards_per_layer_calculation_l3766_376663


namespace NUMINAMATH_CALUDE_total_distance_in_land_miles_l3766_376640

/-- Represents the speed of the sailboat in knots -/
structure SailboatSpeed where
  oneSail : ℝ
  twoSails : ℝ

/-- Represents the travel time in hours -/
structure TravelTime where
  oneSail : ℝ
  twoSails : ℝ

/-- Conversion factors -/
def knotToNauticalMile : ℝ := 1
def nauticalMileToLandMile : ℝ := 1.15

theorem total_distance_in_land_miles 
  (speed : SailboatSpeed) 
  (time : TravelTime) 
  (h1 : speed.oneSail = 25)
  (h2 : speed.twoSails = 50)
  (h3 : time.oneSail = 4)
  (h4 : time.twoSails = 4) :
  (speed.oneSail * time.oneSail + speed.twoSails * time.twoSails) * 
  knotToNauticalMile * nauticalMileToLandMile = 345 := by
  sorry

#check total_distance_in_land_miles

end NUMINAMATH_CALUDE_total_distance_in_land_miles_l3766_376640


namespace NUMINAMATH_CALUDE_soda_price_increase_l3766_376678

theorem soda_price_increase (initial_total : ℝ) (new_candy_price new_soda_price : ℝ) 
  (candy_increase : ℝ) :
  initial_total = 16 →
  new_candy_price = 10 →
  new_soda_price = 12 →
  candy_increase = 0.25 →
  (new_soda_price / (initial_total - new_candy_price / (1 + candy_increase)) - 1) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_increase_l3766_376678


namespace NUMINAMATH_CALUDE_f_not_in_third_quadrant_l3766_376689

/-- The quadratic function under consideration -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- A point is in the third quadrant if both its x and y coordinates are negative -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Theorem stating that the graph of f does not pass through the third quadrant -/
theorem f_not_in_third_quadrant :
  ∀ x : ℝ, ¬(in_third_quadrant x (f x)) :=
sorry

end NUMINAMATH_CALUDE_f_not_in_third_quadrant_l3766_376689


namespace NUMINAMATH_CALUDE_pool_fill_time_l3766_376668

/-- Proves that a pool with the given specifications takes 25 hours to fill -/
theorem pool_fill_time (pool_volume : ℝ) (hose1_rate : ℝ) (hose2_rate : ℝ) 
  (h_volume : pool_volume = 15000)
  (h_hose1 : hose1_rate = 2)
  (h_hose2 : hose2_rate = 3) : 
  pool_volume / (2 * hose1_rate + 2 * hose2_rate) / 60 = 25 := by
  sorry

end NUMINAMATH_CALUDE_pool_fill_time_l3766_376668


namespace NUMINAMATH_CALUDE_integer_solution_system_l3766_376665

theorem integer_solution_system (a b c : ℤ) : 
  a^2 = b*c + 1 ∧ b^2 = a*c + 1 ↔ 
  (a = 1 ∧ b = 0 ∧ c = -1) ∨
  (a = 1 ∧ b = -1 ∧ c = 0) ∨
  (a = -1 ∧ b = 0 ∧ c = 1) ∨
  (a = -1 ∧ b = 1 ∧ c = 0) := by
sorry

end NUMINAMATH_CALUDE_integer_solution_system_l3766_376665


namespace NUMINAMATH_CALUDE_vector_magnitude_equation_l3766_376670

theorem vector_magnitude_equation (k : ℝ) : 
  ‖k • (⟨3, -4⟩ : ℝ × ℝ) + ⟨5, -6⟩‖ = 5 * Real.sqrt 5 ↔ k = 17/25 ∨ k = -19/5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_equation_l3766_376670


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3766_376680

/-- Given an arithmetic sequence {a_n} with a_1 = -2 and S_3 = 0, 
    where S_n is the sum of the first n terms, 
    prove that the common difference is 2. -/
theorem arithmetic_sequence_common_difference : 
  ∀ (a : ℕ → ℚ) (S : ℕ → ℚ),
  (∀ n, S n = (n : ℚ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →  -- Definition of S_n
  a 1 = -2 →                                                     -- a_1 = -2
  S 3 = 0 →                                                      -- S_3 = 0
  a 2 - a 1 = 2 :=                                               -- Common difference is 2
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3766_376680


namespace NUMINAMATH_CALUDE_mikes_tire_changes_l3766_376609

/-- The number of tires changed by Mike in a day -/
def total_tires_changed (
  motorcycles cars bicycles trucks atvs : ℕ)
  (motorcycle_wheels car_wheels bicycle_wheels truck_wheels atv_wheels : ℕ) : ℕ :=
  motorcycles * motorcycle_wheels +
  cars * car_wheels +
  bicycles * bicycle_wheels +
  trucks * truck_wheels +
  atvs * atv_wheels

/-- Theorem stating the total number of tires changed by Mike in a day -/
theorem mikes_tire_changes :
  total_tires_changed 12 10 8 5 7 2 4 2 18 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_mikes_tire_changes_l3766_376609


namespace NUMINAMATH_CALUDE_men_at_first_stop_l3766_376697

/-- Represents the number of people on a subway --/
structure SubwayPopulation where
  women : ℕ
  men : ℕ

/-- The subway population after the first stop --/
def first_stop : SubwayPopulation → Prop
  | ⟨w, m⟩ => m = w - 17

/-- The change in subway population at the second stop --/
def second_stop (pop : SubwayPopulation) : ℕ := 
  pop.women + pop.men + (57 + 18 - 44)

/-- The theorem stating the number of men who got on at the first stop --/
theorem men_at_first_stop (pop : SubwayPopulation) : 
  first_stop pop → second_stop pop = 502 → pop.men = 227 := by
  sorry

end NUMINAMATH_CALUDE_men_at_first_stop_l3766_376697


namespace NUMINAMATH_CALUDE_binary_10101_is_21_l3766_376656

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + (if bit then 2^i else 0)) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_is_21_l3766_376656


namespace NUMINAMATH_CALUDE_prob_neither_correct_l3766_376622

/-- Given probabilities for answering questions correctly, calculate the probability of answering neither correctly -/
theorem prob_neither_correct (P_A P_B P_AB : ℝ) 
  (h1 : P_A = 0.65)
  (h2 : P_B = 0.55)
  (h3 : P_AB = 0.40)
  (h4 : 0 ≤ P_A ∧ P_A ≤ 1)
  (h5 : 0 ≤ P_B ∧ P_B ≤ 1)
  (h6 : 0 ≤ P_AB ∧ P_AB ≤ 1) :
  1 - (P_A + P_B - P_AB) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_prob_neither_correct_l3766_376622


namespace NUMINAMATH_CALUDE_butterfly_ratio_is_three_to_one_l3766_376695

/-- The ratio of time a butterfly spends as a larva to the time spent in a cocoon -/
def butterfly_development_ratio (total_time cocoon_time : ℕ) : ℚ :=
  (total_time - cocoon_time : ℚ) / cocoon_time

/-- Theorem stating that for a butterfly with 120 days total development time and 30 days in cocoon,
    the ratio of time spent as a larva to time in cocoon is 3:1 -/
theorem butterfly_ratio_is_three_to_one :
  butterfly_development_ratio 120 30 = 3 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_ratio_is_three_to_one_l3766_376695


namespace NUMINAMATH_CALUDE_art_class_gender_difference_l3766_376643

theorem art_class_gender_difference (total_students : ℕ) 
  (boy_ratio girl_ratio : ℕ) (h1 : total_students = 42) 
  (h2 : boy_ratio = 3) (h3 : girl_ratio = 4) : 
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    boy_ratio * girls = girl_ratio * boys ∧ 
    girls - boys = 6 := by
sorry

end NUMINAMATH_CALUDE_art_class_gender_difference_l3766_376643


namespace NUMINAMATH_CALUDE_vector_perpendicular_problem_l3766_376626

theorem vector_perpendicular_problem (x : ℝ) : 
  let a : ℝ × ℝ := (-2, x)
  let b : ℝ × ℝ := (1, Real.sqrt 3)
  (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 = 0 → x = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_problem_l3766_376626


namespace NUMINAMATH_CALUDE_expected_total_audience_l3766_376605

theorem expected_total_audience (saturday_attendance : ℕ) 
  (monday_attendance : ℕ) (wednesday_attendance : ℕ) (friday_attendance : ℕ) 
  (actual_total : ℕ) (expected_total : ℕ) : 
  saturday_attendance = 80 →
  monday_attendance = saturday_attendance - 20 →
  wednesday_attendance = monday_attendance + 50 →
  friday_attendance = saturday_attendance + monday_attendance →
  actual_total = saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance →
  actual_total = expected_total + 40 →
  expected_total = 350 := by
sorry

end NUMINAMATH_CALUDE_expected_total_audience_l3766_376605


namespace NUMINAMATH_CALUDE_installation_cost_calculation_l3766_376638

/-- Calculates the installation cost given the purchase details of a refrigerator. -/
theorem installation_cost_calculation
  (purchase_price_after_discount : ℝ)
  (discount_rate : ℝ)
  (transport_cost : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price_after_discount = 12500)
  (h2 : discount_rate = 0.20)
  (h3 : transport_cost = 125)
  (h4 : selling_price = 18400)
  (h5 : selling_price = 1.15 * (purchase_price_after_discount + transport_cost + installation_cost)) :
  installation_cost = 3375 :=
by sorry


end NUMINAMATH_CALUDE_installation_cost_calculation_l3766_376638


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3766_376692

theorem imaginary_part_of_z (z : ℂ) : z = (2 * Complex.I^2 + 4) / (Complex.I + 1) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3766_376692


namespace NUMINAMATH_CALUDE_prob_objects_meet_l3766_376604

/-- The number of steps required for objects to meet -/
def stepsToMeet : ℕ := 9

/-- The possible x-coordinates of meeting points -/
def meetingPoints : List ℕ := [0, 2, 4, 6, 8]

/-- Calculate binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Calculate the number of paths to a meeting point for each object -/
def pathsToPoint (i : ℕ) : ℕ × ℕ :=
  (binomial stepsToMeet i, binomial stepsToMeet (i + 1))

/-- Calculate the probability of meeting at a specific point -/
def probMeetAtPoint (i : ℕ) : ℚ :=
  let (a, b) := pathsToPoint i
  (a * b : ℚ) / (2^(2 * stepsToMeet) : ℚ)

/-- The main theorem: probability of objects meeting -/
theorem prob_objects_meet :
  (meetingPoints.map probMeetAtPoint).sum = 207 / 262144 := by sorry

end NUMINAMATH_CALUDE_prob_objects_meet_l3766_376604


namespace NUMINAMATH_CALUDE_full_house_count_l3766_376601

theorem full_house_count :
  let n_values : ℕ := 13
  let cards_per_value : ℕ := 4
  let full_house_count := n_values * (n_values - 1) * (cards_per_value.choose 3) * (cards_per_value.choose 2)
  full_house_count = 3744 :=
by sorry

end NUMINAMATH_CALUDE_full_house_count_l3766_376601


namespace NUMINAMATH_CALUDE_hamster_lifespan_l3766_376682

theorem hamster_lifespan (fish_lifespan dog_lifespan hamster_lifespan : ℝ) 
  (h1 : fish_lifespan = dog_lifespan + 2)
  (h2 : dog_lifespan = 4 * hamster_lifespan)
  (h3 : fish_lifespan = 12) : 
  hamster_lifespan = 2.5 := by
sorry

end NUMINAMATH_CALUDE_hamster_lifespan_l3766_376682


namespace NUMINAMATH_CALUDE_locus_of_concyclic_points_l3766_376611

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the special points of the triangle
def H : ℝ × ℝ := sorry  -- Orthocenter
def I : ℝ × ℝ := sorry  -- Incenter
def G : ℝ × ℝ := sorry  -- Centroid

-- Define points E and F that divide AB into three equal parts
def E : ℝ × ℝ := sorry
def F : ℝ × ℝ := sorry

-- Define the angle at vertex C
def angle_C : ℝ := sorry

-- Define a predicate for points being concyclic
def are_concyclic (p q r s : ℝ × ℝ) : Prop := sorry

-- Define a predicate for a point being on a circular arc
def on_circular_arc (p center : ℝ × ℝ) (start_point end_point : ℝ × ℝ) (arc_angle : ℝ) : Prop := sorry

theorem locus_of_concyclic_points :
  are_concyclic A B H I →
  (angle_C = 60) ∧ 
  (∃ (center : ℝ × ℝ), on_circular_arc G center E F 120) :=
sorry

end NUMINAMATH_CALUDE_locus_of_concyclic_points_l3766_376611


namespace NUMINAMATH_CALUDE_negation_equivalence_l3766_376636

-- Define a type for polyhedra
structure Polyhedron where
  faces : Set Face

-- Define a type for faces
inductive Face
  | Triangle
  | Quadrilateral
  | Pentagon
  | Other

-- Define the original proposition
def original_proposition : Prop :=
  ∀ p : Polyhedron, ∃ f ∈ p.faces, f = Face.Triangle ∨ f = Face.Quadrilateral ∨ f = Face.Pentagon

-- Define the negation
def negation : Prop :=
  ∃ p : Polyhedron, ∀ f ∈ p.faces, f ≠ Face.Triangle ∧ f ≠ Face.Quadrilateral ∧ f ≠ Face.Pentagon

-- Theorem stating the equivalence
theorem negation_equivalence : ¬original_proposition ↔ negation := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3766_376636


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l3766_376602

theorem no_prime_roots_for_quadratic : ¬∃ k : ℤ, ∃ p q : ℕ,
  Prime p ∧ Prime q ∧ p + q = 95 ∧ p * q = k :=
sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l3766_376602


namespace NUMINAMATH_CALUDE_angle_BEC_measure_l3766_376655

-- Define the geometric configuration
structure GeometricConfig where
  A : Real
  D : Real
  F : Real
  BEC_exists : Bool
  E_above_C : Bool

-- Define the theorem
theorem angle_BEC_measure (config : GeometricConfig) 
  (h1 : config.A = 45)
  (h2 : config.D = 50)
  (h3 : config.F = 55)
  (h4 : config.BEC_exists = true)
  (h5 : config.E_above_C = true) :
  ∃ (BEC : Real), BEC = 10 := by
  sorry

end NUMINAMATH_CALUDE_angle_BEC_measure_l3766_376655


namespace NUMINAMATH_CALUDE_larry_wins_probability_l3766_376631

theorem larry_wins_probability (larry_prob julius_prob : ℚ) : 
  larry_prob = 3/5 →
  julius_prob = 2/5 →
  let win_prob := larry_prob / (1 - (1 - larry_prob) * (1 - julius_prob))
  win_prob = 11/15 := by
sorry

end NUMINAMATH_CALUDE_larry_wins_probability_l3766_376631


namespace NUMINAMATH_CALUDE_inlet_pipe_rate_l3766_376647

/-- Given a tank with specified capacity and emptying times, calculate the inlet pipe rate -/
theorem inlet_pipe_rate (tank_capacity : ℝ) (outlet_time : ℝ) (combined_time : ℝ) :
  tank_capacity = 3200 →
  outlet_time = 5 →
  combined_time = 8 →
  (tank_capacity / combined_time - tank_capacity / outlet_time) * (1 / 60) = 4 := by
  sorry

end NUMINAMATH_CALUDE_inlet_pipe_rate_l3766_376647


namespace NUMINAMATH_CALUDE_five_is_integer_l3766_376651

-- Define the set of natural numbers
def NaturalNumber : Type := ℕ

-- Define the set of integers
def Integer : Type := ℤ

-- Define the property that all natural numbers are integers
axiom all_naturals_are_integers : ∀ (n : NaturalNumber), Integer

-- Define that 5 is a natural number
axiom five_is_natural : NaturalNumber

-- Theorem to prove
theorem five_is_integer : Integer :=
sorry

end NUMINAMATH_CALUDE_five_is_integer_l3766_376651


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l3766_376698

theorem sqrt_expression_equality : 
  (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) + (Real.sqrt 2 - Real.sqrt 3)^2 = 4 - 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l3766_376698


namespace NUMINAMATH_CALUDE_problem_statement_l3766_376699

theorem problem_statement (a b c : ℝ) 
  (h1 : -10 ≤ a) (h2 : a < 0) 
  (h3 : 0 < a) (h4 : a < b) (h5 : b < c) : 
  (a * c < b * c) ∧ (a + c < b + c) ∧ (c / a > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3766_376699


namespace NUMINAMATH_CALUDE_angle_A_is_obtuse_l3766_376615

/-- Triangle ABC with vertices A(2,1), B(-1,4), and C(5,3) -/
structure Triangle where
  A : ℝ × ℝ := (2, 1)
  B : ℝ × ℝ := (-1, 4)
  C : ℝ × ℝ := (5, 3)

/-- Calculate the squared distance between two points -/
def squaredDistance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Check if an angle is obtuse using the cosine law -/
def isObtuse (a b c : ℝ) : Prop :=
  a^2 > b^2 + c^2

theorem angle_A_is_obtuse (t : Triangle) : 
  isObtuse (squaredDistance t.B t.C) (squaredDistance t.A t.B) (squaredDistance t.A t.C) :=
sorry

end NUMINAMATH_CALUDE_angle_A_is_obtuse_l3766_376615


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3766_376612

/-- Given a regular pentagon and a rectangle with perimeters of 75 inches,
    where the rectangle's length is twice its width, prove that the ratio of
    the side length of the pentagon to the width of the rectangle is 6/5. -/
theorem pentagon_rectangle_ratio :
  ∀ (pentagon_side : ℚ) (rect_width : ℚ),
    -- Pentagon perimeter is 75 inches
    5 * pentagon_side = 75 →
    -- Rectangle perimeter is 75 inches, and length is twice the width
    2 * (rect_width + 2 * rect_width) = 75 →
    -- The ratio of pentagon side to rectangle width is 6/5
    pentagon_side / rect_width = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3766_376612


namespace NUMINAMATH_CALUDE_abs_fraction_inequality_l3766_376653

theorem abs_fraction_inequality (x : ℝ) (h : x ≠ 2) :
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_abs_fraction_inequality_l3766_376653


namespace NUMINAMATH_CALUDE_car_a_speed_l3766_376684

/-- Proves that Car A's speed is 58 mph given the problem conditions -/
theorem car_a_speed (initial_distance : ℝ) (car_b_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  initial_distance = 30 →
  car_b_speed = 50 →
  time = 4.75 →
  final_distance = 8 →
  ∃ (car_a_speed : ℝ),
    car_a_speed * time = car_b_speed * time + initial_distance + final_distance ∧
    car_a_speed = 58 := by
  sorry

end NUMINAMATH_CALUDE_car_a_speed_l3766_376684


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3766_376644

theorem rectangle_perimeter (width : ℝ) (length : ℝ) :
  width > 0 →
  length > 0 →
  length = 2 * width →
  width * length = 576 →
  2 * (width + length) = 72 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3766_376644


namespace NUMINAMATH_CALUDE_nissan_cars_sold_l3766_376600

theorem nissan_cars_sold (total_cars : ℕ) (audi_percent : ℚ) (toyota_percent : ℚ) (acura_percent : ℚ) (bmw_percent : ℚ) 
  (h1 : total_cars = 250)
  (h2 : audi_percent = 10 / 100)
  (h3 : toyota_percent = 25 / 100)
  (h4 : acura_percent = 15 / 100)
  (h5 : bmw_percent = 18 / 100)
  : ℕ :=
by
  sorry

#check nissan_cars_sold

end NUMINAMATH_CALUDE_nissan_cars_sold_l3766_376600


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3766_376632

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, x + a * y = 2 → 2 * x + 4 * y = 5 → (1 : ℝ) * (2 : ℝ) + a * (4 : ℝ) = 0) → 
  a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3766_376632


namespace NUMINAMATH_CALUDE_peanut_butter_calories_value_l3766_376635

/-- The number of calories in a serving of peanut butter -/
def peanut_butter_calories : ℕ := sorry

/-- The number of calories in a piece of bread -/
def bread_calories : ℕ := 100

/-- The total number of calories for breakfast -/
def total_calories : ℕ := 500

/-- The number of servings of peanut butter -/
def peanut_butter_servings : ℕ := 2

/-- The number of pieces of bread -/
def bread_pieces : ℕ := 1

theorem peanut_butter_calories_value : 
  bread_calories * bread_pieces + peanut_butter_calories * peanut_butter_servings = total_calories ∧ 
  peanut_butter_calories = 200 := by sorry

end NUMINAMATH_CALUDE_peanut_butter_calories_value_l3766_376635


namespace NUMINAMATH_CALUDE_concert_ticket_sales_l3766_376688

theorem concert_ticket_sales
  (student_price : ℕ)
  (non_student_price : ℕ)
  (total_revenue : ℕ)
  (student_tickets : ℕ)
  (h1 : student_price = 9)
  (h2 : non_student_price = 11)
  (h3 : total_revenue = 20960)
  (h4 : student_tickets = 520) :
  ∃ (non_student_tickets : ℕ),
    student_tickets * student_price + non_student_tickets * non_student_price = total_revenue ∧
    student_tickets + non_student_tickets = 2000 :=
by
  sorry

#check concert_ticket_sales

end NUMINAMATH_CALUDE_concert_ticket_sales_l3766_376688


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l3766_376661

/-- Given a store's pricing strategy and profit margin, 
    prove the initial markup percentage. -/
theorem initial_markup_percentage 
  (initial_cost : ℝ) 
  (markup_percentage : ℝ) 
  (new_year_markup : ℝ) 
  (february_discount : ℝ) 
  (february_profit : ℝ) 
  (h1 : new_year_markup = 0.25) 
  (h2 : february_discount = 0.09) 
  (h3 : february_profit = 0.365) : 
  1.365 = (1 + markup_percentage) * 1.25 * 0.91 := by
  sorry

#check initial_markup_percentage

end NUMINAMATH_CALUDE_initial_markup_percentage_l3766_376661


namespace NUMINAMATH_CALUDE_three_sets_sum_18_with_6_l3766_376628

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem three_sets_sum_18_with_6 :
  (Finset.filter (fun s : Finset ℕ => 
    s.card = 3 ∧ 
    s ⊆ S ∧ 
    6 ∈ s ∧ 
    s.sum id = 18
  ) (S.powerset)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_sets_sum_18_with_6_l3766_376628


namespace NUMINAMATH_CALUDE_class_size_problem_l3766_376645

theorem class_size_problem (x : ℕ) : 
  (40 * x + 60 * 28) / (x + 28) = 54 → x = 12 := by sorry

end NUMINAMATH_CALUDE_class_size_problem_l3766_376645


namespace NUMINAMATH_CALUDE_increasing_order_x_y_z_l3766_376686

theorem increasing_order_x_y_z (x : ℝ) (hx : 1.1 < x ∧ x < 1.2) :
  x < x^x ∧ x^x < x^(x^x) := by
  sorry

end NUMINAMATH_CALUDE_increasing_order_x_y_z_l3766_376686


namespace NUMINAMATH_CALUDE_volume_of_prism_with_inscribed_sphere_l3766_376621

/-- A regular triangular prism with an inscribed sphere -/
structure RegularTriangularPrism where
  -- The radius of the inscribed sphere
  sphere_radius : ℝ
  -- Assertion that the sphere is inscribed in the prism
  sphere_inscribed : sphere_radius > 0

/-- The volume of a regular triangular prism with an inscribed sphere -/
def prism_volume (p : RegularTriangularPrism) : ℝ :=
  -- Definition of volume calculation
  sorry

/-- Theorem: The volume of a regular triangular prism with an inscribed sphere of radius 2 is 48√3 -/
theorem volume_of_prism_with_inscribed_sphere :
  ∀ (p : RegularTriangularPrism), p.sphere_radius = 2 → prism_volume p = 48 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_prism_with_inscribed_sphere_l3766_376621


namespace NUMINAMATH_CALUDE_two_dogs_food_consumption_l3766_376633

/-- The amount of dog food eaten by two dogs per day -/
def total_dog_food (dog1_food : ℝ) (dog2_food : ℝ) : ℝ :=
  dog1_food + dog2_food

/-- Theorem stating that two dogs eating 0.125 scoops each consume 0.25 scoops in total -/
theorem two_dogs_food_consumption :
  total_dog_food 0.125 0.125 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_two_dogs_food_consumption_l3766_376633


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3766_376648

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - x*y = 0) :
  x + 2*y ≥ 8 ∧ (x + 2*y = 8 ↔ x = 2*y) :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3766_376648


namespace NUMINAMATH_CALUDE_woman_birth_year_l3766_376675

theorem woman_birth_year (x : ℕ) (h1 : x > 0) (h2 : x^2 < 1950) 
  (h3 : x^2 + x ≤ 2000) : x^2 = 1936 := by
  sorry

end NUMINAMATH_CALUDE_woman_birth_year_l3766_376675


namespace NUMINAMATH_CALUDE_descendant_divisibility_l3766_376624

theorem descendant_divisibility (N : ℕ) (h : N ≥ 10000 ∧ N < 100000) :
  N % 271 = 0 → (N * 10 + N / 10000 - (N / 10000) * 100000) % 271 = 0 := by
  sorry

end NUMINAMATH_CALUDE_descendant_divisibility_l3766_376624


namespace NUMINAMATH_CALUDE_sin_330_degrees_l3766_376679

theorem sin_330_degrees : Real.sin (330 * π / 180) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l3766_376679


namespace NUMINAMATH_CALUDE_tanya_plums_l3766_376642

/-- The number of plums Tanya bought at the grocery store -/
def plums : ℕ := 6

/-- The total number of pears, apples, and pineapples Tanya bought -/
def other_fruits : ℕ := 12

/-- The number of fruits remaining in the bag after half fell out -/
def remaining_fruits : ℕ := 9

theorem tanya_plums :
  plums = remaining_fruits * 2 - other_fruits :=
by sorry

end NUMINAMATH_CALUDE_tanya_plums_l3766_376642


namespace NUMINAMATH_CALUDE_function_properties_unique_proportional_function_l3766_376677

/-- A proportional function passing through the point (3, 6) -/
def f (x : ℝ) : ℝ := 2 * x

/-- Theorem stating the properties of the function f -/
theorem function_properties :
  (f 3 = 6) ∧
  (f 4 ≠ -2) ∧
  (f (-1.5) ≠ 3) := by
  sorry

/-- Theorem proving that f is the unique proportional function passing through (3, 6) -/
theorem unique_proportional_function (g : ℝ → ℝ) (h : ∀ x, g x = k * x) :
  g 3 = 6 → g = f := by
  sorry

end NUMINAMATH_CALUDE_function_properties_unique_proportional_function_l3766_376677


namespace NUMINAMATH_CALUDE_equation_unique_solution_l3766_376610

theorem equation_unique_solution :
  ∃! x : ℝ, (8 : ℝ)^(2*x+1) * (2 : ℝ)^(3*x+5) = (4 : ℝ)^(3*x+2) ∧ x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_unique_solution_l3766_376610


namespace NUMINAMATH_CALUDE_samson_schedule_solution_l3766_376639

/-- Utility function -/
def utility (math : ℝ) (frisbee : ℝ) : ℝ := 2 * math * frisbee

/-- Wednesday's utility -/
def wednesday_utility (s : ℝ) : ℝ := utility (10 - 2*s) s

/-- Thursday's utility -/
def thursday_utility (s : ℝ) : ℝ := utility (2*s + 4) (3 - s)

/-- The theorem stating that s = 2 is the unique solution -/
theorem samson_schedule_solution :
  ∃! s : ℝ, wednesday_utility s = thursday_utility s ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_samson_schedule_solution_l3766_376639


namespace NUMINAMATH_CALUDE_novelist_writing_speed_l3766_376616

/-- Calculates the effective writing speed given total words, total hours, and break hours -/
def effectiveWritingSpeed (totalWords : ℕ) (totalHours : ℕ) (breakHours : ℕ) : ℕ :=
  totalWords / (totalHours - breakHours)

/-- Proves that the effective writing speed for the given conditions is 750 words per hour -/
theorem novelist_writing_speed :
  effectiveWritingSpeed 60000 100 20 = 750 := by
  sorry

end NUMINAMATH_CALUDE_novelist_writing_speed_l3766_376616


namespace NUMINAMATH_CALUDE_clinton_shoes_count_l3766_376662

/-- Clinton's wardrobe inventory problem -/
theorem clinton_shoes_count :
  ∀ (shoes belts hats : ℕ),
  shoes = 2 * belts →
  belts = hats + 2 →
  hats = 5 →
  shoes = 14 :=
by sorry

end NUMINAMATH_CALUDE_clinton_shoes_count_l3766_376662


namespace NUMINAMATH_CALUDE_inequalities_hold_l3766_376669

theorem inequalities_hold (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) : 
  (m - Real.sqrt m ≥ -1/4) ∧ 
  (1/2 * (m + n)^2 + 1/4 * (m + n) ≥ m * Real.sqrt n + n * Real.sqrt m) := by
  sorry

#check inequalities_hold

end NUMINAMATH_CALUDE_inequalities_hold_l3766_376669


namespace NUMINAMATH_CALUDE_congruence_unique_solution_l3766_376641

theorem congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -1212 [ZMOD 10] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_congruence_unique_solution_l3766_376641


namespace NUMINAMATH_CALUDE_star_emilio_sum_difference_l3766_376630

def star_list := List.range 30

def emilio_list := star_list.map (fun n => 
  if n % 10 = 3 then n - 1
  else if n ≥ 30 then n - 10
  else n)

theorem star_emilio_sum_difference :
  (star_list.sum - emilio_list.sum) = 13 := by sorry

end NUMINAMATH_CALUDE_star_emilio_sum_difference_l3766_376630


namespace NUMINAMATH_CALUDE_quadratic_vertex_l3766_376623

/-- The quadratic function f(x) = -3(x+2)^2 + 1 has vertex coordinates (-2, 1) -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := fun x ↦ -3 * (x + 2)^2 + 1
  (∀ x, f x ≤ f (-2)) ∧ f (-2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l3766_376623


namespace NUMINAMATH_CALUDE_smallest_set_with_both_progressions_l3766_376618

/-- A sequence of integers forms a geometric progression of length 5 -/
def IsGeometricProgression (s : Finset ℤ) : Prop :=
  ∃ (a q : ℤ), q ≠ 0 ∧ s = {a, a*q, a*q^2, a*q^3, a*q^4}

/-- A sequence of integers forms an arithmetic progression of length 5 -/
def IsArithmeticProgression (s : Finset ℤ) : Prop :=
  ∃ (a d : ℤ), s = {a, a+d, a+2*d, a+3*d, a+4*d}

/-- The main theorem stating the smallest number of distinct integers -/
theorem smallest_set_with_both_progressions :
  ∀ (s : Finset ℤ), (∃ (s1 s2 : Finset ℤ), s1 ⊆ s ∧ s2 ⊆ s ∧ 
    IsGeometricProgression s1 ∧ IsArithmeticProgression s2) →
  s.card ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_set_with_both_progressions_l3766_376618


namespace NUMINAMATH_CALUDE_factory_production_average_l3766_376650

theorem factory_production_average (first_25_avg : ℝ) (last_5_avg : ℝ) (total_days : ℕ) :
  first_25_avg = 60 →
  last_5_avg = 48 →
  total_days = 30 →
  (25 * first_25_avg + 5 * last_5_avg) / total_days = 58 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_average_l3766_376650


namespace NUMINAMATH_CALUDE_no_integer_solution_l3766_376666

theorem no_integer_solution : ∀ (x y : ℤ), x^2 + 4*x - 11 ≠ 8*y := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3766_376666


namespace NUMINAMATH_CALUDE_product_xyz_l3766_376664

theorem product_xyz (x y z : ℕ+) 
  (h1 : x + 2 * y = z) 
  (h2 : x^2 - 4 * y^2 + z^2 = 310) : 
  x * y * z = 11935 ∨ x * y * z = 2015 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_l3766_376664


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l3766_376654

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) : ∃ k : ℕ, k ∣ n ∧ k = 12 ∧ ∀ m : ℕ, m ∣ n → m ≤ k := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l3766_376654


namespace NUMINAMATH_CALUDE_oatmeal_cookies_given_away_l3766_376606

/-- Represents the number of cookies in a dozen. -/
def dozen : ℕ := 12

/-- Represents the total number of cookies Ann baked. -/
def totalBaked : ℕ := 3 * dozen + 2 * dozen + 4 * dozen

/-- Represents the number of sugar cookies Ann gave away. -/
def sugarGivenAway : ℕ := (3 * dozen) / 2

/-- Represents the number of chocolate chip cookies Ann gave away. -/
def chocolateGivenAway : ℕ := (5 * dozen) / 2

/-- Represents the number of cookies Ann kept. -/
def cookiesKept : ℕ := 36

/-- Proves that Ann gave away 2 dozen oatmeal raisin cookies. -/
theorem oatmeal_cookies_given_away :
  ∃ (x : ℕ), x * dozen + sugarGivenAway + chocolateGivenAway + cookiesKept = totalBaked ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_oatmeal_cookies_given_away_l3766_376606


namespace NUMINAMATH_CALUDE_boat_round_trip_time_l3766_376674

/-- Calculates the total time for a round trip by boat given the boat's speed in standing water,
    the stream's speed, and the distance to the destination. -/
theorem boat_round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 15)
  (h2 : stream_speed = 3)
  (h3 : distance = 180)
  : (distance / (boat_speed + stream_speed)) + (distance / (boat_speed - stream_speed)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_boat_round_trip_time_l3766_376674


namespace NUMINAMATH_CALUDE_triangular_region_area_ratio_l3766_376672

/-- Represents a square divided into a 6x6 grid -/
structure GridSquare where
  side_length : ℝ
  grid_size : ℕ := 6

/-- Represents the triangular region in the GridSquare -/
structure TriangularRegion (gs : GridSquare) where
  vertex1 : ℝ × ℝ  -- Midpoint of one side
  vertex2 : ℝ × ℝ  -- Diagonal corner of 4x4 block
  vertex3 : ℝ × ℝ  -- Midpoint of adjacent side

/-- Calculates the area of the GridSquare -/
def area_grid_square (gs : GridSquare) : ℝ :=
  gs.side_length ^ 2

/-- Calculates the area of the TriangularRegion -/
noncomputable def area_triangular_region (gs : GridSquare) (tr : TriangularRegion gs) : ℝ :=
  sorry  -- Actual calculation would go here

/-- The main theorem stating the ratio of areas -/
theorem triangular_region_area_ratio (gs : GridSquare) (tr : TriangularRegion gs) :
  area_triangular_region gs tr / area_grid_square gs = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_triangular_region_area_ratio_l3766_376672


namespace NUMINAMATH_CALUDE_nathan_total_earnings_l3766_376683

/-- Nathan's hourly wage in dollars -/
def hourly_wage : ℝ := 6

/-- Hours worked in the second week of July -/
def hours_week2 : ℝ := 12

/-- Hours worked in the third week of July -/
def hours_week3 : ℝ := 18

/-- Earnings difference between the third and second week -/
def earnings_difference : ℝ := 36

theorem nathan_total_earnings : 
  hourly_wage * hours_week2 + hourly_wage * hours_week3 = 180 := by
  sorry

end NUMINAMATH_CALUDE_nathan_total_earnings_l3766_376683


namespace NUMINAMATH_CALUDE_half_minus_third_equals_sixth_l3766_376687

theorem half_minus_third_equals_sixth : (1/2 : ℚ) - (1/3 : ℚ) = 1/6 := by sorry

end NUMINAMATH_CALUDE_half_minus_third_equals_sixth_l3766_376687


namespace NUMINAMATH_CALUDE_secant_theorem_l3766_376673

-- Define the basic geometric elements
variable (A B C M A₁ B₁ C₁ : ℝ × ℝ)

-- Define the triangle ABC
def is_triangle (A B C : ℝ × ℝ) : Prop := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define that M is not on the sides or extensions of ABC
def M_not_on_triangle (A B C M : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, M ≠ A + t • (B - A) ∧ 
           M ≠ B + t • (C - B) ∧ 
           M ≠ C + t • (A - C)

-- Define the secant through M intersecting sides (or extensions) at A₁, B₁, C₁
def secant_intersects (A B C M A₁ B₁ C₁ : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ t₃ : ℝ, 
    A₁ = A + t₁ • (B - A) ∧
    B₁ = B + t₂ • (C - B) ∧
    C₁ = C + t₃ • (A - C) ∧
    (∃ s₁ s₂ s₃ : ℝ, M = A₁ + s₁ • (B₁ - A₁) ∧
                     M = B₁ + s₂ • (C₁ - B₁) ∧
                     M = C₁ + s₃ • (A₁ - C₁))

-- Define oriented area function
noncomputable def oriented_area (P Q R : ℝ × ℝ) : ℝ := sorry

-- Define oriented distance function
noncomputable def oriented_distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem secant_theorem (A B C M A₁ B₁ C₁ : ℝ × ℝ) 
  (h_triangle : is_triangle A B C)
  (h_M_not_on : M_not_on_triangle A B C M)
  (h_secant : secant_intersects A B C M A₁ B₁ C₁) :
  (oriented_area A B M) / (oriented_distance M C₁) + 
  (oriented_area B C M) / (oriented_distance M A₁) + 
  (oriented_area C A M) / (oriented_distance M B₁) = 0 := by sorry

end NUMINAMATH_CALUDE_secant_theorem_l3766_376673


namespace NUMINAMATH_CALUDE_crosswalk_wait_probability_l3766_376660

/-- Represents the duration of the red light in seconds -/
def red_light_duration : ℕ := 40

/-- Represents the minimum waiting time in seconds for which we calculate the probability -/
def min_wait_time : ℕ := 15

/-- The probability of waiting at least 'min_wait_time' seconds for a green light when encountering a red light -/
def wait_probability : ℚ := 5/8

/-- Theorem stating that the probability of waiting at least 'min_wait_time' seconds for a green light
    when encountering a red light of duration 'red_light_duration' is equal to 'wait_probability' -/
theorem crosswalk_wait_probability :
  (red_light_duration - min_wait_time : ℚ) / red_light_duration = wait_probability := by
  sorry

end NUMINAMATH_CALUDE_crosswalk_wait_probability_l3766_376660


namespace NUMINAMATH_CALUDE_lindsey_remaining_money_l3766_376617

/-- Calculates Lindsey's remaining money after saving and spending --/
theorem lindsey_remaining_money 
  (september_savings : ℕ) 
  (october_savings : ℕ) 
  (november_savings : ℕ) 
  (mom_bonus_threshold : ℕ) 
  (mom_bonus : ℕ) 
  (video_game_cost : ℕ) : 
  (september_savings = 50 ∧ 
   october_savings = 37 ∧ 
   november_savings = 11 ∧ 
   mom_bonus_threshold = 75 ∧ 
   mom_bonus = 25 ∧ 
   video_game_cost = 87) → 
  (let total_savings := september_savings + october_savings + november_savings
   let total_with_bonus := total_savings + (if total_savings > mom_bonus_threshold then mom_bonus else 0)
   let remaining_money := total_with_bonus - video_game_cost
   remaining_money = 36) := by
sorry

end NUMINAMATH_CALUDE_lindsey_remaining_money_l3766_376617


namespace NUMINAMATH_CALUDE_total_players_specific_l3766_376629

/-- The number of players in a sports event with overlapping groups --/
def totalPlayers (kabadi khoKho soccer kabadi_khoKho kabadi_soccer khoKho_soccer all_three : ℕ) : ℕ :=
  kabadi + khoKho + soccer - kabadi_khoKho - kabadi_soccer - khoKho_soccer + all_three

/-- Theorem stating the total number of players given the specific conditions --/
theorem total_players_specific : totalPlayers 50 80 30 15 10 25 8 = 118 := by
  sorry

end NUMINAMATH_CALUDE_total_players_specific_l3766_376629


namespace NUMINAMATH_CALUDE_rectangle_area_l3766_376603

theorem rectangle_area (p : ℝ) (p_small : ℝ) (h1 : p = 30) (h2 : p_small = 16) :
  let w := (p - p_small) / 2
  let l := p_small / 2 - w + w
  w * l = 56 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3766_376603


namespace NUMINAMATH_CALUDE_sphere_box_height_l3766_376619

/-- A rectangular box with a large sphere and eight smaller spheres -/
structure SphereBox where
  h : ℝ
  large_sphere_radius : ℝ
  small_sphere_radius : ℝ
  box_width : ℝ
  box_length : ℝ
  num_small_spheres : ℕ

/-- The configuration of spheres in the box satisfies the given conditions -/
def valid_configuration (box : SphereBox) : Prop :=
  box.large_sphere_radius = 3 ∧
  box.small_sphere_radius = 1.5 ∧
  box.box_width = 6 ∧
  box.box_length = 6 ∧
  box.num_small_spheres = 8 ∧
  ∀ (small_sphere : Fin box.num_small_spheres),
    (∃ (side1 side2 side3 : ℝ), 
      side1 + side2 + side3 = box.box_width + box.box_length + box.h) ∧
    (box.large_sphere_radius + box.small_sphere_radius = 
      box.box_width / 2 - box.small_sphere_radius)

/-- The height of the box is 9 given the valid configuration -/
theorem sphere_box_height (box : SphereBox) :
  valid_configuration box → box.h = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_box_height_l3766_376619


namespace NUMINAMATH_CALUDE_common_divisors_9180_10080_l3766_376607

theorem common_divisors_9180_10080 : 
  let a := 9180
  let b := 10080
  (a % 7 = 0) → 
  (b % 7 = 0) → 
  (Finset.filter (fun d => d ∣ a ∧ d ∣ b) (Finset.range (min a b + 1))).card = 36 := by
sorry

end NUMINAMATH_CALUDE_common_divisors_9180_10080_l3766_376607


namespace NUMINAMATH_CALUDE_honda_cars_in_chennai_l3766_376627

def total_cars : ℕ := 900
def red_honda_percentage : ℚ := 90 / 100
def total_red_percentage : ℚ := 60 / 100
def red_non_honda_percentage : ℚ := 225 / 1000

theorem honda_cars_in_chennai :
  ∃ (h : ℕ), h = 500 ∧
  (h : ℚ) * red_honda_percentage + (total_cars - h : ℚ) * red_non_honda_percentage = (total_cars : ℚ) * total_red_percentage :=
by sorry

end NUMINAMATH_CALUDE_honda_cars_in_chennai_l3766_376627


namespace NUMINAMATH_CALUDE_unique_solution_l3766_376649

theorem unique_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (6 - y) = 9)
  (eq2 : y * (6 - z) = 9)
  (eq3 : z * (6 - x) = 9) :
  x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l3766_376649


namespace NUMINAMATH_CALUDE_sum_is_composite_l3766_376608

theorem sum_is_composite (a b c d : ℕ+) 
  (h : a^2 - a*b + b^2 = c^2 - c*d + d^2) : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + b + c + d = x * y :=
sorry

end NUMINAMATH_CALUDE_sum_is_composite_l3766_376608


namespace NUMINAMATH_CALUDE_difference_is_895_l3766_376671

/-- The smallest positive three-digit integer congruent to 7 (mod 13) -/
def m : ℕ := sorry

/-- The smallest positive four-digit integer congruent to 7 (mod 13) -/
def n : ℕ := sorry

/-- m is a three-digit number -/
axiom m_three_digit : 100 ≤ m ∧ m < 1000

/-- n is a four-digit number -/
axiom n_four_digit : 1000 ≤ n ∧ n < 10000

/-- m is congruent to 7 (mod 13) -/
axiom m_congruence : m % 13 = 7

/-- n is congruent to 7 (mod 13) -/
axiom n_congruence : n % 13 = 7

/-- m is the smallest such number -/
axiom m_smallest : ∀ k : ℕ, 100 ≤ k ∧ k < 1000 ∧ k % 13 = 7 → m ≤ k

/-- n is the smallest such number -/
axiom n_smallest : ∀ k : ℕ, 1000 ≤ k ∧ k < 10000 ∧ k % 13 = 7 → n ≤ k

theorem difference_is_895 : n - m = 895 := by sorry

end NUMINAMATH_CALUDE_difference_is_895_l3766_376671


namespace NUMINAMATH_CALUDE_smallest_valid_n_l3766_376694

def is_valid (n : ℕ) : Prop :=
  n > 1 ∧ ∀ p, Nat.Prime p → p ∣ n → n > 1200 * p

theorem smallest_valid_n :
  is_valid 3888 ∧ ∀ m, m < 3888 → ¬is_valid m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l3766_376694


namespace NUMINAMATH_CALUDE_eventually_constant_l3766_376685

/-- The set of positive integers -/
def PositiveInts : Set ℕ := {n : ℕ | n > 0}

/-- The winning set for (n,S)-nim game -/
def winning_set (S : Set ℕ) : Set ℕ :=
  {n : ℕ | ∃ (strategy : ℕ → ℕ), ∀ (m : ℕ), m < n → strategy m ∈ S ∧ strategy m ≤ n}

/-- The function f that maps a set S to its winning set -/
def f (S : Set ℕ) : Set ℕ := winning_set S

/-- Iterate f k times -/
def iterate_f (S : Set ℕ) : ℕ → Set ℕ
  | 0 => S
  | k + 1 => f (iterate_f S k)

/-- The main theorem: the sequence of iterations of f eventually becomes constant -/
theorem eventually_constant (T : Set ℕ) : ∃ (k : ℕ), iterate_f T k = iterate_f T (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_eventually_constant_l3766_376685


namespace NUMINAMATH_CALUDE_max_visible_sum_l3766_376659

-- Define a cube type
structure Cube :=
  (faces : Fin 6 → Nat)

-- Define the block type
structure Block :=
  (cubes : Fin 4 → Cube)

-- Function to calculate the sum of visible faces
def sumVisibleFaces (b : Block) : Nat :=
  sorry

-- Theorem statement
theorem max_visible_sum :
  ∃ (b : Block),
    (∀ i : Fin 6, ∀ c : Fin 4, 1 ≤ (b.cubes c).faces i ∧ (b.cubes c).faces i ≤ 6) ∧
    (∀ c1 c2 : Fin 4, c1 ≠ c2 → ∃ i j : Fin 6, (b.cubes c1).faces i = (b.cubes c2).faces j) ∧
    (sumVisibleFaces b = 68) ∧
    (∀ b' : Block, sumVisibleFaces b' ≤ sumVisibleFaces b) :=
  sorry


end NUMINAMATH_CALUDE_max_visible_sum_l3766_376659


namespace NUMINAMATH_CALUDE_disco_vote_participants_l3766_376634

theorem disco_vote_participants :
  ∀ (initial_voters : ℕ) 
    (initial_oct22_percent initial_oct29_percent : ℚ)
    (additional_voters : ℕ)
    (final_oct29_percent : ℚ),
  initial_oct22_percent + initial_oct29_percent = 1 →
  initial_oct22_percent = 35 / 100 →
  initial_oct29_percent = 65 / 100 →
  additional_voters = 80 →
  final_oct29_percent = 45 / 100 →
  initial_oct29_percent * initial_voters = 
    final_oct29_percent * (initial_voters + additional_voters) →
  initial_voters + additional_voters = 260 := by
sorry


end NUMINAMATH_CALUDE_disco_vote_participants_l3766_376634


namespace NUMINAMATH_CALUDE_parabola_translation_l3766_376613

/-- The equation of a parabola translated upwards by 1 unit from y = x^2 -/
theorem parabola_translation (x y : ℝ) : 
  (y = x^2) → (∃ y', y' = y + 1 ∧ y' = x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3766_376613


namespace NUMINAMATH_CALUDE_hiking_distance_proof_l3766_376676

theorem hiking_distance_proof (total_distance : ℝ) : 
  (total_distance / 3 + (2 * total_distance / 3) / 3 + (4 * total_distance / 9) / 4 + 24 = total_distance) →
  (total_distance = 72 ∧ 
   total_distance / 3 = 24 ∧ 
   (2 * total_distance / 3) / 3 = 16 ∧ 
   (4 * total_distance / 9) / 4 = 8) :=
by
  sorry

#check hiking_distance_proof

end NUMINAMATH_CALUDE_hiking_distance_proof_l3766_376676


namespace NUMINAMATH_CALUDE_basketball_win_percentage_l3766_376691

theorem basketball_win_percentage (total_games : ℕ) (first_games : ℕ) (first_wins : ℕ) (target_percentage : ℚ) : 
  total_games = 110 →
  first_games = 60 →
  first_wins = 45 →
  target_percentage = 3/4 →
  ∃ (remaining_wins : ℕ), 
    remaining_wins = 38 ∧ 
    (first_wins + remaining_wins : ℚ) / total_games = target_percentage :=
by sorry

end NUMINAMATH_CALUDE_basketball_win_percentage_l3766_376691
