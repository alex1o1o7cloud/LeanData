import Mathlib

namespace total_yen_calculation_l2806_280689

theorem total_yen_calculation (checking_account savings_account : ℕ) 
  (h1 : checking_account = 6359)
  (h2 : savings_account = 3485) :
  checking_account + savings_account = 9844 := by
  sorry

end total_yen_calculation_l2806_280689


namespace heidi_to_danielle_ratio_l2806_280636

/-- The number of rooms in Danielle's apartment -/
def danielles_rooms : ℕ := 6

/-- The number of rooms in Grant's apartment -/
def grants_rooms : ℕ := 2

/-- The ratio of Grant's rooms to Heidi's rooms -/
def grant_to_heidi_ratio : ℚ := 1 / 9

/-- The number of rooms in Heidi's apartment -/
def heidis_rooms : ℕ := grants_rooms * 9

theorem heidi_to_danielle_ratio : 
  (heidis_rooms : ℚ) / danielles_rooms = 3 / 1 := by
  sorry

end heidi_to_danielle_ratio_l2806_280636


namespace circle_centers_distance_l2806_280668

theorem circle_centers_distance (r₁ r₂ : ℝ) (angle : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 95) (h₃ : angle = 60) :
  let distance := 2 * r₂ - 2 * r₁
  distance = 160 :=
sorry

end circle_centers_distance_l2806_280668


namespace right_angle_sufficiency_not_necessity_l2806_280633

theorem right_angle_sufficiency_not_necessity (A B C : ℝ) :
  -- Triangle ABC exists
  (0 < A) ∧ (0 < B) ∧ (0 < C) ∧ (A + B + C = π) →
  -- 1. If angle C is 90°, then cos A + sin A = cos B + sin B
  (C = π / 2 → Real.cos A + Real.sin A = Real.cos B + Real.sin B) ∧
  -- 2. There exists a triangle where cos A + sin A = cos B + sin B, but angle C ≠ 90°
  ∃ (A' B' C' : ℝ), (0 < A') ∧ (0 < B') ∧ (0 < C') ∧ (A' + B' + C' = π) ∧
    (Real.cos A' + Real.sin A' = Real.cos B' + Real.sin B') ∧ (C' ≠ π / 2) :=
by sorry

end right_angle_sufficiency_not_necessity_l2806_280633


namespace positive_integer_solutions_l2806_280615

theorem positive_integer_solutions : 
  ∀ x y z : ℕ+, 
    (x + y = z ∧ x^2 * y = z^2 + 1) ↔ 
    ((x = 5 ∧ y = 2 ∧ z = 7) ∨ (x = 5 ∧ y = 3 ∧ z = 8)) :=
by sorry

end positive_integer_solutions_l2806_280615


namespace ant_problem_l2806_280669

/-- Represents the number of ants for each species on Day 0 -/
structure AntCounts where
  a : ℕ  -- Species A
  b : ℕ  -- Species B
  c : ℕ  -- Species C

/-- Calculates the total number of ants on a given day -/
def totalAnts (day : ℕ) (counts : AntCounts) : ℕ :=
  2^day * counts.a + 3^day * counts.b + 4^day * counts.c

theorem ant_problem (counts : AntCounts) :
  totalAnts 0 counts = 50 →
  totalAnts 4 counts = 6561 →
  4^4 * counts.c = 5632 := by
  sorry

end ant_problem_l2806_280669


namespace largest_a_value_l2806_280652

theorem largest_a_value : 
  ∀ a : ℝ, (3*a + 4)*(a - 2) = 9*a → a ≤ 4 :=
by sorry

end largest_a_value_l2806_280652


namespace strawberry_to_fruit_ratio_l2806_280656

-- Define the total garden size
def garden_size : ℕ := 64

-- Define the fruit section size (half of the garden)
def fruit_section : ℕ := garden_size / 2

-- Define the strawberry section size
def strawberry_section : ℕ := 8

-- Theorem to prove the ratio of strawberry section to fruit section
theorem strawberry_to_fruit_ratio :
  (strawberry_section : ℚ) / fruit_section = 1 / 4 := by
  sorry

end strawberry_to_fruit_ratio_l2806_280656


namespace parallel_vectors_theta_l2806_280644

theorem parallel_vectors_theta (θ : Real) 
  (h1 : θ > 0) (h2 : θ < Real.pi / 2)
  (a : Fin 2 → Real) (b : Fin 2 → Real)
  (ha : a = ![3/2, Real.sin θ])
  (hb : b = ![Real.cos θ, 1/3])
  (h_parallel : ∃ (k : Real), a = k • b) :
  θ = Real.pi / 4 := by
sorry

end parallel_vectors_theta_l2806_280644


namespace angle_in_third_quadrant_l2806_280640

theorem angle_in_third_quadrant (α : Real) : 
  (Real.sin α * Real.tan α < 0) → 
  (Real.cos α / Real.tan α < 0) → 
  (α > Real.pi ∧ α < 3 * Real.pi / 2) := by
sorry

end angle_in_third_quadrant_l2806_280640


namespace profit_percentage_cricket_bat_l2806_280693

/-- The profit percentage calculation for a cricket bat sale -/
theorem profit_percentage_cricket_bat (selling_price profit : ℝ)
  (h1 : selling_price = 850)
  (h2 : profit = 230) :
  ∃ (percentage : ℝ), abs (percentage - 37.10) < 0.01 ∧
  percentage = (profit / (selling_price - profit)) * 100 := by
  sorry

end profit_percentage_cricket_bat_l2806_280693


namespace function_symmetry_implies_a_equals_four_l2806_280676

/-- Given a quadratic function f(x) = 2x^2 - ax + 3, 
    if f(1-x) = f(1+x) for all real x, then a = 4 -/
theorem function_symmetry_implies_a_equals_four (a : ℝ) : 
  (∀ x : ℝ, 2*(1-x)^2 - a*(1-x) + 3 = 2*(1+x)^2 - a*(1+x) + 3) → 
  a = 4 := by
  sorry

end function_symmetry_implies_a_equals_four_l2806_280676


namespace equation_in_y_l2806_280698

theorem equation_in_y (x y : ℝ) 
  (eq1 : 3 * x^2 - 5 * x + 4 * y + 6 = 0)
  (eq2 : 3 * x - 2 * y + 1 = 0) :
  4 * y^2 - 2 * y + 24 = 0 := by
  sorry

end equation_in_y_l2806_280698


namespace water_amount_l2806_280686

/-- The number of boxes -/
def num_boxes : ℕ := 10

/-- The number of bottles in each box -/
def bottles_per_box : ℕ := 50

/-- The capacity of each bottle in liters -/
def bottle_capacity : ℚ := 12

/-- The fraction of the bottle's capacity that is filled -/
def fill_fraction : ℚ := 3/4

/-- The total amount of water in liters contained in all boxes -/
def total_water : ℚ := num_boxes * bottles_per_box * bottle_capacity * fill_fraction

theorem water_amount : total_water = 4500 := by
  sorry

end water_amount_l2806_280686


namespace remainder_not_always_power_of_four_l2806_280691

theorem remainder_not_always_power_of_four :
  ∃ n : ℕ, n ≥ 2 ∧ ∃ k : ℕ, (2^(2^n) : ℕ) % (2^n - 1) = k ∧ ¬∃ m : ℕ, k = 4^m := by
  sorry

end remainder_not_always_power_of_four_l2806_280691


namespace rectangle_perimeter_l2806_280679

/-- A rectangle with length thrice its breadth and area 75 square meters has a perimeter of 40 meters. -/
theorem rectangle_perimeter (breadth : ℝ) (length : ℝ) (area : ℝ) (perimeter : ℝ) : 
  length = 3 * breadth →
  area = 75 →
  area = length * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 40 := by
sorry

end rectangle_perimeter_l2806_280679


namespace complex_magnitude_l2806_280648

theorem complex_magnitude (i : ℂ) (h : i * i = -1) :
  Complex.abs (i + 2 * i^2 + 3 * i^3) = 2 * Real.sqrt 2 := by
  sorry

end complex_magnitude_l2806_280648


namespace simplest_quadratic_radical_l2806_280622

theorem simplest_quadratic_radical :
  let a := Real.sqrt 8
  let b := Real.sqrt 7
  let c := Real.sqrt 12
  let d := Real.sqrt (1/3)
  (∃ (x y : ℝ), a = x * Real.sqrt y ∧ x ≠ 1) ∧
  (∃ (x y : ℝ), c = x * Real.sqrt y ∧ x ≠ 1) ∧
  (∃ (x y : ℝ), d = x * Real.sqrt y ∧ x ≠ 1) ∧
  (∀ (x y : ℝ), b = x * Real.sqrt y → x = 1) :=
by sorry

end simplest_quadratic_radical_l2806_280622


namespace upstream_distance_is_18_l2806_280653

/-- Represents the swimming scenario with given conditions -/
structure SwimmingScenario where
  still_speed : ℝ  -- Speed of the man in still water (km/h)
  downstream_distance : ℝ  -- Distance swam downstream (km)
  downstream_time : ℝ  -- Time spent swimming downstream (hours)
  upstream_time : ℝ  -- Time spent swimming upstream (hours)

/-- Calculates the upstream distance given a swimming scenario -/
def upstream_distance (s : SwimmingScenario) : ℝ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating that for the given conditions, the upstream distance is 18 km -/
theorem upstream_distance_is_18 :
  let s : SwimmingScenario := {
    still_speed := 11.5,
    downstream_distance := 51,
    downstream_time := 3,
    upstream_time := 3
  }
  upstream_distance s = 18 := by
  sorry

end upstream_distance_is_18_l2806_280653


namespace stream_speed_l2806_280688

/-- Proves that given a boat's travel times and distances upstream and downstream, the speed of the stream is 1 km/h -/
theorem stream_speed (b : ℝ) (s : ℝ) 
  (h1 : (b + s) * 10 = 100) 
  (h2 : (b - s) * 25 = 200) : 
  s = 1 := by
  sorry

end stream_speed_l2806_280688


namespace exists_monochromatic_congruent_triangle_l2806_280628

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring of the plane
def Coloring := Point → Color

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Define congruence for triangles
def Congruent (t1 t2 : Triangle) : Prop := sorry

-- Define the property of having all vertices of the same color
def SameColor (t : Triangle) (coloring : Coloring) : Prop :=
  coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c

-- The main theorem
theorem exists_monochromatic_congruent_triangle :
  ∃ (T : Triangle), ∀ (coloring : Coloring),
    ∃ (T' : Triangle), Congruent T T' ∧ SameColor T' coloring := by sorry

end exists_monochromatic_congruent_triangle_l2806_280628


namespace real_number_classification_l2806_280665

theorem real_number_classification :
  Set.univ = {x : ℝ | x > 0} ∪ {x : ℝ | x < 0} ∪ {(0 : ℝ)} := by sorry

end real_number_classification_l2806_280665


namespace apples_per_basket_l2806_280681

theorem apples_per_basket (baskets_per_tree : ℕ) (trees : ℕ) (total_apples : ℕ) :
  baskets_per_tree = 20 →
  trees = 10 →
  total_apples = 3000 →
  total_apples / (trees * baskets_per_tree) = 15 := by
  sorry

end apples_per_basket_l2806_280681


namespace quadratic_equation_solution_l2806_280637

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 + 4*x₁ = 5) ∧ 
  (x₂^2 + 4*x₂ = 5) ∧ 
  x₁ = 1 ∧ 
  x₂ = -5 := by
  sorry

end quadratic_equation_solution_l2806_280637


namespace abs_neg_three_l2806_280602

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end abs_neg_three_l2806_280602


namespace value_of_x_l2806_280630

theorem value_of_x (x y z : ℚ) : 
  x = (1 / 3) * y → 
  y = (1 / 10) * z → 
  z = 100 → 
  x = 10 / 3 := by
  sorry

end value_of_x_l2806_280630


namespace least_integer_square_72_more_than_double_l2806_280617

theorem least_integer_square_72_more_than_double :
  ∃ (x : ℤ), x^2 = 2*x + 72 ∧ ∀ (y : ℤ), y^2 = 2*y + 72 → x ≤ y :=
by sorry

end least_integer_square_72_more_than_double_l2806_280617


namespace money_split_ratio_l2806_280666

/-- Given two people splitting money in a ratio of 2:3, where the smaller share is $50,
    prove that the total amount shared is $125. -/
theorem money_split_ratio (parker_share richie_share total : ℕ) : 
  parker_share = 50 →
  parker_share + richie_share = total →
  2 * richie_share = 3 * parker_share →
  total = 125 := by
sorry

end money_split_ratio_l2806_280666


namespace goldbach_multiplication_counterexample_l2806_280625

theorem goldbach_multiplication_counterexample :
  ∃ p : ℕ, Prime p ∧ p > 5 ∧
  (∀ q r : ℕ, Prime q → Prime r → Odd q → Odd r → p ≠ q * r) ∧
  (∀ q : ℕ, Prime q → Odd q → p ≠ q^2) := by
  sorry

end goldbach_multiplication_counterexample_l2806_280625


namespace johns_phone_bill_l2806_280660

/-- Calculates the total phone bill given the monthly fee, per-minute rate, and minutes used. -/
def total_bill (monthly_fee : ℝ) (per_minute_rate : ℝ) (minutes_used : ℝ) : ℝ :=
  monthly_fee + per_minute_rate * minutes_used

/-- Theorem stating that John's phone bill is $12.02 given the specified conditions. -/
theorem johns_phone_bill :
  let monthly_fee : ℝ := 5
  let per_minute_rate : ℝ := 0.25
  let minutes_used : ℝ := 28.08
  total_bill monthly_fee per_minute_rate minutes_used = 12.02 := by
sorry


end johns_phone_bill_l2806_280660


namespace factorization_proof_l2806_280611

theorem factorization_proof (a m n : ℝ) : a * m^2 - 2 * a * m * n + a * n^2 = a * (m - n)^2 := by
  sorry

end factorization_proof_l2806_280611


namespace cos_2015_eq_neg_sin_55_l2806_280608

theorem cos_2015_eq_neg_sin_55 (m : ℝ) (h : Real.sin (55 * π / 180) = m) :
  Real.cos (2015 * π / 180) = -m := by
  sorry

end cos_2015_eq_neg_sin_55_l2806_280608


namespace painted_cubes_count_l2806_280663

/-- A cube-based construction with 5 layers -/
structure CubeConstruction where
  middle_layer : Nat
  other_layers : Nat
  unpainted_cubes : Nat

/-- The number of cubes with at least one face painted in the construction -/
def painted_cubes (c : CubeConstruction) : Nat :=
  c.middle_layer + 4 * c.other_layers - c.unpainted_cubes

/-- Theorem: In the given cube construction, 104 cubes have at least one face painted -/
theorem painted_cubes_count (c : CubeConstruction) 
  (h1 : c.middle_layer = 16)
  (h2 : c.other_layers = 24)
  (h3 : c.unpainted_cubes = 8) : 
  painted_cubes c = 104 := by
  sorry

end painted_cubes_count_l2806_280663


namespace function_bounds_l2806_280607

theorem function_bounds (k : ℕ) (f : ℕ → ℕ) 
  (h_increasing : ∀ m n, m < n → f m < f n)
  (h_composition : ∀ n, f (f n) = k * n) :
  ∀ n : ℕ, (2 * k : ℚ) / (k + 1 : ℚ) * n ≤ f n ∧ (f n : ℚ) ≤ (k + 1 : ℚ) / 2 * n :=
by sorry

end function_bounds_l2806_280607


namespace cube_and_square_root_problem_l2806_280612

theorem cube_and_square_root_problem (a b : ℝ) :
  (2*b - 2*a)^(1/3) = -2 →
  (4*a + 3*b)^(1/2) = 3 →
  (a = 3 ∧ b = -1) ∧ ((5*a - b)^(1/2) = 4 ∨ (5*a - b)^(1/2) = -4) :=
by sorry

end cube_and_square_root_problem_l2806_280612


namespace distance_between_points_l2806_280647

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-3.5, -4.5)
  let p2 : ℝ × ℝ := (3.5, 2.5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 7 * Real.sqrt 2 := by
  sorry

end distance_between_points_l2806_280647


namespace triangle_angle_range_l2806_280694

theorem triangle_angle_range (A B C : Real) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) (h5 : Real.log (Real.tan A) + Real.log (Real.tan C) = 2 * Real.log (Real.tan B)) : 
  π / 3 ≤ B ∧ B < π / 2 := by
sorry

end triangle_angle_range_l2806_280694


namespace strawberry_weight_difference_l2806_280614

theorem strawberry_weight_difference (marco_weight dad_weight total_weight : ℕ) 
  (h1 : marco_weight = 30)
  (h2 : total_weight = 47)
  (h3 : total_weight = marco_weight + dad_weight) :
  marco_weight - dad_weight = 13 := by
  sorry

end strawberry_weight_difference_l2806_280614


namespace percentage_problem_l2806_280673

theorem percentage_problem (N P : ℝ) : 
  N = 50 → 
  N = (P / 100) * N + 40 → 
  P = 20 := by sorry

end percentage_problem_l2806_280673


namespace min_exponent_sum_l2806_280687

theorem min_exponent_sum (h : ℕ+) (a b c : ℕ+) 
  (h_div_225 : 225 ∣ h)
  (h_div_216 : 216 ∣ h)
  (h_factorization : h = 2^(a:ℕ) * 3^(b:ℕ) * 5^(c:ℕ)) :
  a + b + c ≥ 8 ∧ ∃ (h' : ℕ+) (a' b' c' : ℕ+), 
    225 ∣ h' ∧ 216 ∣ h' ∧ h' = 2^(a':ℕ) * 3^(b':ℕ) * 5^(c':ℕ) ∧ a' + b' + c' = 8 :=
by sorry

end min_exponent_sum_l2806_280687


namespace quadruple_theorem_l2806_280624

def is_valid_quadruple (a b c d : ℝ) : Prop :=
  (a = b * c ∨ a = b * d ∨ a = c * d) ∧
  (b = a * c ∨ b = a * d ∨ b = c * d) ∧
  (c = a * b ∨ c = a * d ∨ c = b * d) ∧
  (d = a * b ∨ d = a * c ∨ d = b * c)

def is_solution_quadruple (a b c d : ℝ) : Prop :=
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  ((a = 1 ∧ b = 1 ∧ c = -1 ∧ d = -1) ∨
   (a = 1 ∧ b = -1 ∧ c = 1 ∧ d = -1) ∨
   (a = 1 ∧ b = -1 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = 1 ∧ c = 1 ∧ d = -1) ∨
   (a = -1 ∧ b = 1 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = -1 ∧ c = 1 ∧ d = 1)) ∨
  ((a = 1 ∧ b = -1 ∧ c = -1 ∧ d = -1) ∨
   (a = -1 ∧ b = 1 ∧ c = -1 ∧ d = -1) ∨
   (a = -1 ∧ b = -1 ∧ c = 1 ∧ d = -1) ∨
   (a = -1 ∧ b = -1 ∧ c = -1 ∧ d = 1))

theorem quadruple_theorem (a b c d : ℝ) :
  is_valid_quadruple a b c d → is_solution_quadruple a b c d := by
  sorry

end quadruple_theorem_l2806_280624


namespace decimal_parts_fraction_decimal_parts_fraction_proof_l2806_280683

theorem decimal_parts_fraction : ℝ → Prop := 
  fun x => let a : ℤ := ⌊2 + Real.sqrt 2⌋
           let b : ℝ := 2 + Real.sqrt 2 - a
           let c : ℤ := ⌊4 - Real.sqrt 2⌋
           let d : ℝ := 4 - Real.sqrt 2 - c
           (b + d) / (a * c) = 1/6

theorem decimal_parts_fraction_proof : decimal_parts_fraction 2 := by
  sorry

end decimal_parts_fraction_decimal_parts_fraction_proof_l2806_280683


namespace polynomial_degree_l2806_280621

/-- The degree of the polynomial resulting from the expansion of 
    (3x^5 + 2x^3 - x + 7)(4x^11 - 6x^8 + 5x^5 - 15) - (x^2 + 3)^8 is 16 -/
theorem polynomial_degree : ℕ := by
  sorry

end polynomial_degree_l2806_280621


namespace road_sign_ratio_l2806_280632

/-- Represents the number of road signs at each intersection -/
structure RoadSigns where
  s₁ : ℕ
  s₂ : ℕ
  s₃ : ℕ
  s₄ : ℕ

/-- The conditions of the road sign problem -/
def road_sign_conditions (r : RoadSigns) : Prop :=
  r.s₁ = 40 ∧
  r.s₂ > r.s₁ ∧
  r.s₃ = 2 * r.s₂ ∧
  r.s₄ = r.s₃ - 20 ∧
  r.s₁ + r.s₂ + r.s₃ + r.s₄ = 270

/-- The theorem stating that under the given conditions, 
    the ratio of road signs at the second intersection to the first is 5:4 -/
theorem road_sign_ratio (r : RoadSigns) : 
  road_sign_conditions r → r.s₂ * 4 = r.s₁ * 5 := by
  sorry

end road_sign_ratio_l2806_280632


namespace min_value_quadratic_l2806_280603

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = -5 ∧ ∀ x, x^2 + 2*x - 4 ≥ m := by
  sorry

end min_value_quadratic_l2806_280603


namespace erasers_ratio_l2806_280604

def erasers_problem (hanna rachel tanya_red tanya_total : ℕ) : Prop :=
  hanna = 4 ∧
  tanya_total = 20 ∧
  hanna = 2 * rachel ∧
  rachel = tanya_red / 2 - 3 ∧
  tanya_red ≤ tanya_total

theorem erasers_ratio :
  ∀ hanna rachel tanya_red tanya_total,
    erasers_problem hanna rachel tanya_red tanya_total →
    (tanya_red : ℚ) / tanya_total = 1 / 2 :=
by
  sorry

end erasers_ratio_l2806_280604


namespace intersection_sum_l2806_280697

theorem intersection_sum (a b : ℚ) : 
  (2 = (1/3) * 3 + a) → 
  (3 = (1/5) * 2 + b) → 
  a + b = 18/5 := by
sorry

end intersection_sum_l2806_280697


namespace senior_class_college_attendance_l2806_280639

theorem senior_class_college_attendance 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (percent_not_attending : ℝ) 
  (h1 : num_boys = 300)
  (h2 : num_girls = 240)
  (h3 : percent_not_attending = 0.3)
  : (((1 - percent_not_attending) * (num_boys + num_girls)) / (num_boys + num_girls)) * 100 = 70 := by
  sorry

end senior_class_college_attendance_l2806_280639


namespace eighth_of_2_36_equals_2_33_l2806_280601

theorem eighth_of_2_36_equals_2_33 : ∃ y : ℕ, (1 / 8 : ℝ) * (2 ^ 36) = 2 ^ y → y = 33 := by
  sorry

end eighth_of_2_36_equals_2_33_l2806_280601


namespace books_left_to_read_l2806_280619

theorem books_left_to_read (total_books read_books : ℕ) : 
  total_books = 13 → read_books = 9 → total_books - read_books = 4 := by
  sorry

end books_left_to_read_l2806_280619


namespace correct_propositions_l2806_280684

-- Define the propositions
def proposition1 : Prop := ∀ p q : Prop, (¬(p ∨ q)) → (¬p ∧ ¬q)

def proposition2 : Prop := 
  (∃ x : ℝ, x^2 + 1 > 3*x) ∧ (∀ x : ℝ, x^2 - 1 < 3*x)

def proposition3 : Prop := 
  ∀ a b m : ℝ, (a < b) → (a*m^2 < b*m^2)

def proposition4 : Prop := 
  ∀ p q : Prop, (p → q) ∧ ¬(q → p) → (¬p → ¬q) ∧ ¬(¬q → ¬p)

-- Theorem stating which propositions are correct
theorem correct_propositions : 
  proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4 := by
  sorry

end correct_propositions_l2806_280684


namespace inscribed_square_area_l2806_280646

/-- Given a square with side length a and a circle circumscribed around it,
    the area of a square inscribed in one of the resulting segments is a²/25 -/
theorem inscribed_square_area (a : ℝ) (a_pos : 0 < a) :
  ∃ (x : ℝ), x > 0 ∧ x^2 = a^2 / 25 := by
  sorry

end inscribed_square_area_l2806_280646


namespace triangle_base_length_l2806_280642

/-- Given a triangle with area 24.36 and height 5.8, its base length is 8.4 -/
theorem triangle_base_length : 
  ∀ (base : ℝ), 
    (24.36 = (base * 5.8) / 2) → 
    base = 8.4 := by
  sorry

end triangle_base_length_l2806_280642


namespace basketball_only_count_l2806_280654

theorem basketball_only_count (total students_basketball students_table_tennis students_neither : ℕ) 
  (h1 : total = 30)
  (h2 : students_basketball = 15)
  (h3 : students_table_tennis = 10)
  (h4 : students_neither = 8)
  (h5 : total = students_basketball + students_table_tennis - students_both + students_neither)
  (students_both : ℕ) :
  students_basketball - students_both = 12 := by
  sorry

end basketball_only_count_l2806_280654


namespace intersection_of_sets_l2806_280651

theorem intersection_of_sets (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
  let B : Set ℝ := {x | x^2 + 2*(a + 1)*x + (a^2 - 5) = 0}
  (A ∩ B = {2}) → (a = -1 ∨ a = -3) := by
sorry

end intersection_of_sets_l2806_280651


namespace medium_sized_fir_trees_l2806_280626

theorem medium_sized_fir_trees (total : ℕ) (oaks : ℕ) (saplings : ℕ) 
  (h1 : total = 96) 
  (h2 : oaks = 15) 
  (h3 : saplings = 58) : 
  total - oaks - saplings = 23 := by
  sorry

end medium_sized_fir_trees_l2806_280626


namespace max_value_of_expression_l2806_280645

def is_distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_nonzero_digit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

def expression (a b c : ℕ) : ℚ :=
  1 / (a + 2010 / (b + 1 / c))

theorem max_value_of_expression :
  ∀ a b c : ℕ,
    is_distinct a b c →
    is_nonzero_digit a →
    is_nonzero_digit b →
    is_nonzero_digit c →
    expression a b c ≤ 1 / 203 :=
by sorry

end max_value_of_expression_l2806_280645


namespace selection_theorem_l2806_280692

/-- The number of students in the group -/
def total_students : Nat := 6

/-- The number of students to be selected -/
def selected_students : Nat := 4

/-- The number of subjects -/
def subjects : Nat := 4

/-- The number of students who cannot participate in a specific subject -/
def restricted_students : Nat := 2

/-- The number of different selection plans -/
def selection_plans : Nat := 240

theorem selection_theorem :
  (total_students.factorial / (total_students - selected_students).factorial) -
  (restricted_students * ((total_students - 1).factorial / (total_students - selected_students).factorial)) =
  selection_plans := by
  sorry

end selection_theorem_l2806_280692


namespace employee_pay_l2806_280623

theorem employee_pay (total : ℝ) (x y : ℝ) (h1 : total = 572) (h2 : x + y = total) (h3 : x = 1.2 * y) : y = 260 := by
  sorry

end employee_pay_l2806_280623


namespace subset_union_equality_l2806_280655

theorem subset_union_equality (n : ℕ+) (A : Fin (n + 1) → Set (Fin n)) 
  (h : ∀ i, (A i).Nonempty) :
  ∃ (I J : Set (Fin (n + 1))), I.Nonempty ∧ J.Nonempty ∧ I ∩ J = ∅ ∧
    (⋃ i ∈ I, A i) = (⋃ j ∈ J, A j) := by
  sorry

end subset_union_equality_l2806_280655


namespace solution_set_when_a_is_one_range_of_a_containing_interval_l2806_280696

-- Define the functions f and g
def f (a x : ℝ) : ℝ := -x^2 + a*x + 4
def g (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem for part 1
theorem solution_set_when_a_is_one :
  let a := 1
  ∃ S : Set ℝ, S = {x | f a x ≥ g x} ∧ 
    S = Set.Icc (-1) (((-1 : ℝ) + Real.sqrt 17) / 2) :=
sorry

-- Theorem for part 2
theorem range_of_a_containing_interval :
  ∃ R : Set ℝ, R = {a | ∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ g x} ∧
    R = Set.Icc (-1) 1 :=
sorry

end solution_set_when_a_is_one_range_of_a_containing_interval_l2806_280696


namespace intersection_of_M_and_N_l2806_280680

-- Define the sets M and N
def M : Set ℝ := {x | -4 < x ∧ x < -2}
def N : Set ℝ := {x | x^2 + 5*x + 6 < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -3 < x ∧ x < -2} := by sorry

end intersection_of_M_and_N_l2806_280680


namespace ellipse_focal_property_l2806_280606

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- The left focus of the ellipse -/
def leftFocus : ℝ × ℝ := (-3, 0)

/-- The left directrix of the ellipse -/
def leftDirectrix (x : ℝ) : Prop := x = -25/3

/-- A line passing through the left focus -/
def lineThroughFocus (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 3)

/-- Point D is to the right of the left focus -/
def pointD (a θ : ℝ) : Prop := a > -3

/-- The circle with MN as diameter passes through F₁ -/
def circleCondition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + 3) * (x₂ + 3) + y₁ * y₂ = 0

/-- The main theorem -/
theorem ellipse_focal_property (k a θ : ℝ) (x₁ y₁ x₂ y₂ xₘ xₙ yₘ yₙ : ℝ) :
  ellipse x₁ y₁ →
  ellipse x₂ y₂ →
  lineThroughFocus k x₁ y₁ →
  lineThroughFocus k x₂ y₂ →
  pointD a θ →
  leftDirectrix xₘ →
  leftDirectrix xₙ →
  circleCondition xₘ yₘ xₙ yₙ →
  a = 5 := by
sorry

end ellipse_focal_property_l2806_280606


namespace four_digit_sum_l2806_280638

theorem four_digit_sum (a b c d : Nat) : 
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  (∃ (x y z : Nat), x < 10 ∧ y < 10 ∧ z < 10 ∧
    ((1000 * a + 100 * b + 10 * c + d) + (100 * x + 10 * y + z) = 6031 ∨
     (1000 * a + 100 * b + 10 * c + d) + (100 * a + 10 * y + z) = 6031 ∨
     (1000 * a + 100 * b + 10 * c + d) + (100 * a + 10 * b + z) = 6031 ∨
     (1000 * a + 100 * b + 10 * c + d) + (100 * x + 10 * y + c) = 6031)) →
  a + b + c + d = 20 := by
sorry

end four_digit_sum_l2806_280638


namespace journey_time_ratio_l2806_280667

theorem journey_time_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) 
  (h1 : distance = 180)
  (h2 : original_time = 6)
  (h3 : new_speed = 20)
  : (distance / new_speed) / original_time = 3 / 2 := by
  sorry

#check journey_time_ratio

end journey_time_ratio_l2806_280667


namespace negation_of_exists_greater_l2806_280670

theorem negation_of_exists_greater (p : Prop) :
  (¬ ∃ (n : ℕ), 2^n > 1000) ↔ (∀ (n : ℕ), 2^n ≤ 1000) :=
by sorry

end negation_of_exists_greater_l2806_280670


namespace joanna_reading_speed_l2806_280613

theorem joanna_reading_speed :
  ∀ (total_pages : ℕ) (monday_hours tuesday_hours remaining_hours : ℝ) (pages_per_hour : ℝ),
    total_pages = 248 →
    monday_hours = 3 →
    tuesday_hours = 6.5 →
    remaining_hours = 6 →
    (monday_hours + tuesday_hours + remaining_hours) * pages_per_hour = total_pages →
    pages_per_hour = 16 := by
  sorry

end joanna_reading_speed_l2806_280613


namespace min_value_on_interval_l2806_280677

/-- The function f(x) = x^2 - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The closed interval [2, 3] -/
def I : Set ℝ := Set.Icc 2 3

theorem min_value_on_interval :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ x ∈ I, f x ≥ min_val :=
sorry

end min_value_on_interval_l2806_280677


namespace intersection_points_convex_ngon_l2806_280657

/-- The number of intersection points of the diagonals in a convex n-gon -/
def intersectionPoints (n : ℕ) : ℕ :=
  Nat.choose n 4

/-- Theorem: The number of intersection points of the diagonals in a convex n-gon
    is equal to (n choose 4) for n ≥ 4 -/
theorem intersection_points_convex_ngon (n : ℕ) (h : n ≥ 4) :
  intersectionPoints n = Nat.choose n 4 := by
  sorry

end intersection_points_convex_ngon_l2806_280657


namespace walk_a_thon_miles_difference_l2806_280635

theorem walk_a_thon_miles_difference 
  (last_year_rate : ℝ) 
  (this_year_rate : ℝ) 
  (last_year_total : ℝ) 
  (h1 : last_year_rate = 4)
  (h2 : this_year_rate = 2.75)
  (h3 : last_year_total = 44) :
  (last_year_total / this_year_rate) - (last_year_total / last_year_rate) = 5 :=
by sorry

end walk_a_thon_miles_difference_l2806_280635


namespace water_pump_problem_l2806_280664

theorem water_pump_problem (t₁ t₂ t_combined : ℝ) 
  (h₁ : t₂ = 6)
  (h₂ : t_combined = 3.6)
  (h₃ : 1 / t₁ + 1 / t₂ = 1 / t_combined) :
  t₁ = 9 := by
sorry

end water_pump_problem_l2806_280664


namespace b_value_l2806_280650

/-- The value of b that satisfies the given conditions -/
def find_b : ℝ := sorry

/-- The line equation y = b - x -/
def line_equation (x y : ℝ) : Prop := y = find_b - x

/-- P is the intersection point of the line with the y-axis -/
def P : ℝ × ℝ := (0, find_b)

/-- S is the intersection point of the line with x = 6 -/
def S : ℝ × ℝ := (6, find_b - 6)

/-- Q is the intersection point of the line with the x-axis -/
def Q : ℝ × ℝ := (find_b, 0)

/-- O is the origin -/
def O : ℝ × ℝ := (0, 0)

/-- R is the point (6, 0) -/
def R : ℝ × ℝ := (6, 0)

/-- The area of triangle QRS -/
def area_QRS : ℝ := sorry

/-- The area of triangle QOP -/
def area_QOP : ℝ := sorry

theorem b_value :
  0 < find_b ∧ 
  find_b < 6 ∧ 
  line_equation (P.1) (P.2) ∧
  line_equation (S.1) (S.2) ∧
  (area_QRS / area_QOP = 4 / 25) →
  ∃ ε > 0, |find_b - 4.3| < ε :=
sorry

end b_value_l2806_280650


namespace five_skill_players_wait_l2806_280672

/-- Represents the water cooler scenario for a football team -/
structure WaterCooler where
  totalWater : ℕ
  numLinemen : ℕ
  numSkillPlayers : ℕ
  linemenWater : ℕ
  skillPlayerWater : ℕ

/-- Calculates the number of skill position players who must wait for water -/
def skillPlayersWaiting (wc : WaterCooler) : ℕ :=
  let linemenTotalWater := wc.numLinemen * wc.linemenWater
  let remainingWater := wc.totalWater - linemenTotalWater
  let skillPlayersServed := remainingWater / wc.skillPlayerWater
  wc.numSkillPlayers - skillPlayersServed

/-- Theorem stating that 5 skill position players must wait for water in the given scenario -/
theorem five_skill_players_wait (wc : WaterCooler) 
  (h1 : wc.totalWater = 126)
  (h2 : wc.numLinemen = 12)
  (h3 : wc.numSkillPlayers = 10)
  (h4 : wc.linemenWater = 8)
  (h5 : wc.skillPlayerWater = 6) :
  skillPlayersWaiting wc = 5 := by
  sorry

#eval skillPlayersWaiting { totalWater := 126, numLinemen := 12, numSkillPlayers := 10, linemenWater := 8, skillPlayerWater := 6 }

end five_skill_players_wait_l2806_280672


namespace smallest_third_side_l2806_280674

theorem smallest_third_side (a b : ℝ) (ha : a = 7.5) (hb : b = 11.5) :
  ∃ (s : ℕ), s = 5 ∧ 
  (a + s > b) ∧ (a + b > s) ∧ (b + s > a) ∧
  (∀ (t : ℕ), t < s → ¬(a + t > b ∧ a + b > t ∧ b + t > a)) :=
by
  sorry

end smallest_third_side_l2806_280674


namespace smallest_fraction_between_l2806_280609

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧ 
  (∀ p' q' : ℕ+, (3 : ℚ) / 5 < p' / q' ∧ p' / q' < (5 : ℚ) / 8 → q ≤ q') →
  q - p = 5 := by sorry

end smallest_fraction_between_l2806_280609


namespace theater_revenue_l2806_280605

theorem theater_revenue (n : ℕ) (cost total_revenue actual_revenue : ℝ) :
  (total_revenue = cost * 1.2) →
  (actual_revenue = total_revenue * 0.95) →
  (actual_revenue = cost * 1.14) :=
by sorry

end theater_revenue_l2806_280605


namespace sin_squared_sum_range_l2806_280675

theorem sin_squared_sum_range (α β : ℝ) :
  3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 = 2 * Real.sin α →
  ∃ (x : ℝ), x = (Real.sin α)^2 + (Real.sin β)^2 ∧ 0 ≤ x ∧ x ≤ 4/9 :=
by sorry

end sin_squared_sum_range_l2806_280675


namespace largest_base_8_three_digit_in_base_10_l2806_280627

/-- The largest three-digit number in a given base -/
def largest_three_digit (base : ℕ) : ℕ :=
  (base - 1) * base^2 + (base - 1) * base^1 + (base - 1) * base^0

/-- Theorem: The largest three-digit base-8 number in base-10 is 511 -/
theorem largest_base_8_three_digit_in_base_10 :
  largest_three_digit 8 = 511 := by sorry

end largest_base_8_three_digit_in_base_10_l2806_280627


namespace contradiction_assumption_l2806_280699

theorem contradiction_assumption (a b : ℝ) : 
  (¬(a > b → 3*a > 3*b) ↔ 3*a ≤ 3*b) :=
by sorry

end contradiction_assumption_l2806_280699


namespace intersection_point_property_l2806_280685

theorem intersection_point_property (x₀ : ℝ) (h1 : x₀ ≠ 0) (h2 : Real.tan x₀ = -x₀) :
  (x₀^2 + 1) * (1 + Real.cos (2 * x₀)) = 2 := by
  sorry

end intersection_point_property_l2806_280685


namespace equation_solution_l2806_280659

theorem equation_solution : ∃ x : ℝ, 2*x + 17 = 32 - 3*x ∧ x = 3 := by sorry

end equation_solution_l2806_280659


namespace alice_weekly_distance_l2806_280634

/-- Represents the walking distances for a single day --/
structure DailyWalk where
  morning : ℕ
  evening : ℕ

/-- Alice's walking schedule for the week --/
def aliceSchedule : List DailyWalk := [
  ⟨21, 0⟩,  -- Monday
  ⟨14, 0⟩,  -- Tuesday
  ⟨22, 0⟩,  -- Wednesday
  ⟨19, 0⟩,  -- Thursday
  ⟨20, 0⟩   -- Friday
]

/-- Calculates the total walking distance for a day --/
def totalDailyDistance (day : DailyWalk) : ℕ :=
  day.morning + day.evening

/-- Calculates the total walking distance for the week --/
def totalWeeklyDistance (schedule : List DailyWalk) : ℕ :=
  schedule.map totalDailyDistance |>.sum

/-- Theorem: Alice's total walking distance for the week is 96 miles --/
theorem alice_weekly_distance :
  totalWeeklyDistance aliceSchedule = 96 := by
  sorry

end alice_weekly_distance_l2806_280634


namespace prob_two_girls_from_five_l2806_280658

/-- The probability of selecting 2 girls as representatives from a group of 5 students (2 boys and 3 girls) is 3/10. -/
theorem prob_two_girls_from_five (total : ℕ) (boys : ℕ) (girls : ℕ) (representatives : ℕ) :
  total = 5 →
  boys = 2 →
  girls = 3 →
  representatives = 2 →
  (Nat.choose girls representatives : ℚ) / (Nat.choose total representatives : ℚ) = 3 / 10 := by
sorry

end prob_two_girls_from_five_l2806_280658


namespace cubic_polynomial_factor_l2806_280616

/-- Given a cubic polynomial of the form 3x^3 - dx + 18 with a quadratic factor x^2 + qx + 2,
    prove that d = -6 -/
theorem cubic_polynomial_factor (d : ℝ) : 
  (∃ q : ℝ, ∃ m : ℝ, ∀ x : ℝ, 
    3 * x^3 - d * x + 18 = (x^2 + q * x + 2) * (m * x)) → 
  d = -6 :=
by sorry

end cubic_polynomial_factor_l2806_280616


namespace greeting_cards_exchange_l2806_280661

theorem greeting_cards_exchange (x : ℕ) : x > 0 → x * (x - 1) = 1980 → ∀ (i j : ℕ), i < x ∧ j < x ∧ i ≠ j → ∃ (total : ℕ), total = 1980 ∧ total = x * (x - 1) := by
  sorry

end greeting_cards_exchange_l2806_280661


namespace survey_respondents_count_l2806_280671

theorem survey_respondents_count :
  ∀ (x y : ℕ),
    x = 200 →
    4 * y = x →
    x + y = 250 :=
by sorry

end survey_respondents_count_l2806_280671


namespace book_arrangement_theorem_l2806_280690

/-- The number of ways to arrange books on a shelf --/
def arrange_books (num_math_books : ℕ) (num_history_books : ℕ) : ℕ :=
  let math_book_arrangements := (num_math_books.choose 2) * (2 * 2)
  let history_book_arrangements := num_history_books.factorial
  math_book_arrangements * history_book_arrangements

/-- Theorem: The number of ways to arrange 4 math books and 6 history books with 2 math books on each end is 17280 --/
theorem book_arrangement_theorem :
  arrange_books 4 6 = 17280 := by
  sorry

end book_arrangement_theorem_l2806_280690


namespace number_relations_l2806_280682

theorem number_relations (A B C : ℝ) : 
  A - B = 1860 ∧ 
  0.075 * A = 0.125 * B ∧ 
  0.15 * B = 0.05 * C → 
  A = 4650 ∧ B = 2790 ∧ C = 8370 := by
sorry

end number_relations_l2806_280682


namespace f_greater_g_iff_a_geq_half_l2806_280643

noncomputable section

open Real

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a - log x

def g (x : ℝ) : ℝ := 1/x - Real.exp 1 / (Real.exp x)

-- State the theorem
theorem f_greater_g_iff_a_geq_half (a : ℝ) :
  (∀ x > 1, f a x > g x) ↔ a ≥ 1/2 := by sorry

end

end f_greater_g_iff_a_geq_half_l2806_280643


namespace kayla_apples_l2806_280618

theorem kayla_apples (total : ℕ) (kayla kylie : ℕ) : 
  total = 200 →
  total = kayla + kylie →
  kayla = kylie / 4 →
  kayla = 40 := by
sorry

end kayla_apples_l2806_280618


namespace solution_equals_one_l2806_280662

theorem solution_equals_one (x y : ℝ) 
  (eq1 : 2 * x + y = 4) 
  (eq2 : x + 2 * y = 5) : 
  x = 1 := by
sorry

end solution_equals_one_l2806_280662


namespace expression_evaluation_l2806_280620

theorem expression_evaluation : 
  (0.82 : Real)^3 - (0.1 : Real)^3 / (0.82 : Real)^2 + 0.082 + (0.1 : Real)^2 = 0.641881 := by
  sorry

end expression_evaluation_l2806_280620


namespace temple_shop_charge_l2806_280695

/-- The charge per object at the temple shop -/
def charge_per_object : ℕ → ℕ → ℕ → ℕ → ℚ
  | num_people, shoes_per_person, socks_per_person, mobiles_per_person =>
    let total_objects := num_people * (shoes_per_person + socks_per_person + mobiles_per_person)
    let total_cost := 165
    total_cost / total_objects

theorem temple_shop_charge :
  charge_per_object 3 2 2 1 = 11 := by
  sorry

end temple_shop_charge_l2806_280695


namespace min_value_expression_l2806_280610

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 6) :
  (9 / a + 16 / b + 25 / c) ≥ 18 :=
sorry

end min_value_expression_l2806_280610


namespace factorization_equality_l2806_280631

theorem factorization_equality (a : ℝ) : (a + 2) * (a - 2) - 3 * a = (a - 4) * (a + 1) := by
  sorry

end factorization_equality_l2806_280631


namespace sorting_problem_l2806_280641

/-- The number of parcels sorted by a machine per hour relative to a person -/
def machine_efficiency : ℕ := 20

/-- The number of machines used in the comparison -/
def num_machines : ℕ := 5

/-- The number of people used in the comparison -/
def num_people : ℕ := 20

/-- The number of parcels sorted in the comparison -/
def parcels_sorted : ℕ := 6000

/-- The time difference in hours between machines and people sorting -/
def time_difference : ℕ := 4

/-- The number of hours machines work per day -/
def machine_work_hours : ℕ := 16

/-- The number of parcels that need to be sorted per day -/
def daily_parcels : ℕ := 100000

/-- The number of parcels sorted manually by one person per hour -/
def parcels_per_person (x : ℕ) : Prop :=
  (parcels_sorted / (num_people * x)) - (parcels_sorted / (num_machines * machine_efficiency * x)) = time_difference

/-- The minimum number of machines needed to sort the daily parcels -/
def machines_needed (y : ℕ) : Prop :=
  y = (daily_parcels + machine_work_hours * machine_efficiency * 60 - 1) / (machine_work_hours * machine_efficiency * 60)

theorem sorting_problem :
  ∃ (x y : ℕ), parcels_per_person x ∧ machines_needed y ∧ x = 60 ∧ y = 6 :=
sorry

end sorting_problem_l2806_280641


namespace common_divisors_9240_8000_l2806_280629

theorem common_divisors_9240_8000 : ∃ n : ℕ, n = (Nat.divisors (Nat.gcd 9240 8000)).card ∧ n = 8 := by
  sorry

end common_divisors_9240_8000_l2806_280629


namespace count_valid_triangles_l2806_280600

/-- A point in the 4x4 grid --/
structure GridPoint where
  x : Fin 4
  y : Fin 4

/-- A triangle formed by three grid points --/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Function to check if three points are collinear --/
def collinear (p1 p2 p3 : GridPoint) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Function to check if a triangle has positive area --/
def positiveArea (t : GridTriangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of all possible grid points --/
def allGridPoints : Finset GridPoint :=
  sorry

/-- The set of all possible triangles with positive area --/
def validTriangles : Finset GridTriangle :=
  sorry

/-- The main theorem --/
theorem count_valid_triangles :
  Finset.card validTriangles = 516 :=
sorry

end count_valid_triangles_l2806_280600


namespace subtraction_equivalence_l2806_280649

theorem subtraction_equivalence : 596 - 130 - 270 = 596 - (130 + 270) := by
  sorry

end subtraction_equivalence_l2806_280649


namespace cube_product_divided_l2806_280678

theorem cube_product_divided : (12 : ℝ)^3 * 6^3 / 432 = 864 := by
  sorry

end cube_product_divided_l2806_280678
