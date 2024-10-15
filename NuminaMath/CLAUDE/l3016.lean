import Mathlib

namespace NUMINAMATH_CALUDE_four_digit_divisible_by_9_l3016_301674

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

theorem four_digit_divisible_by_9 (B : ℕ) : 
  B ≤ 9 → is_divisible_by_9 (5000 + 100 * B + 10 * B + 3) → B = 5 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_9_l3016_301674


namespace NUMINAMATH_CALUDE_train_crossing_time_l3016_301657

/-- Proves that a train of given length and speed takes a specific time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 100 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3016_301657


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3016_301668

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = x / (x^2 - 1)) ↔ x ≠ 1 ∧ x ≠ -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3016_301668


namespace NUMINAMATH_CALUDE_multiplication_mistake_l3016_301601

theorem multiplication_mistake (x : ℕ) : x = 43 := by
  have h1 : 136 * x - 1224 = 136 * 34 := by sorry
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_l3016_301601


namespace NUMINAMATH_CALUDE_jia_opened_physical_store_l3016_301630

-- Define the possible shop types
inductive ShopType
| Taobao
| WeChat
| Physical

-- Define the graduates
inductive Graduate
| Jia
| Yi
| Bing

-- Define a function that assigns a shop type to each graduate
def shop : Graduate → ShopType := sorry

-- Define the statements made by each graduate
def jia_statement : Prop :=
  shop Graduate.Jia = ShopType.Taobao ∧ shop Graduate.Yi = ShopType.WeChat

def yi_statement : Prop :=
  shop Graduate.Jia = ShopType.WeChat ∧ shop Graduate.Bing = ShopType.Taobao

def bing_statement : Prop :=
  shop Graduate.Jia = ShopType.Physical ∧ shop Graduate.Yi = ShopType.Taobao

-- Define a function to count the number of true parts in a statement
def true_count (statement : Prop) : Nat := sorry

-- Theorem: Given the conditions, Jia must have opened a physical store
theorem jia_opened_physical_store :
  (true_count jia_statement = 1) →
  (true_count yi_statement = 1) →
  (true_count bing_statement = 1) →
  (shop Graduate.Jia = ShopType.Physical) :=
by sorry

end NUMINAMATH_CALUDE_jia_opened_physical_store_l3016_301630


namespace NUMINAMATH_CALUDE_grocery_receipt_total_cost_l3016_301648

/-- The total cost of three items after applying a tax -/
def totalCostAfterTax (sponge shampoo soap taxRate : ℚ) : ℚ :=
  let preTaxTotal := sponge + shampoo + soap
  let taxAmount := preTaxTotal * taxRate
  preTaxTotal + taxAmount

/-- Theorem stating that the total cost after tax for the given items is $15.75 -/
theorem grocery_receipt_total_cost :
  totalCostAfterTax (420/100) (760/100) (320/100) (5/100) = 1575/100 := by
  sorry

end NUMINAMATH_CALUDE_grocery_receipt_total_cost_l3016_301648


namespace NUMINAMATH_CALUDE_largest_integral_x_l3016_301659

theorem largest_integral_x : ∃ (x : ℤ),
  (1 / 4 : ℚ) < (x : ℚ) / 6 ∧ 
  (x : ℚ) / 6 < (7 / 11 : ℚ) ∧
  (∀ (y : ℤ), (1 / 4 : ℚ) < (y : ℚ) / 6 ∧ (y : ℚ) / 6 < (7 / 11 : ℚ) → y ≤ x) ∧
  x = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_integral_x_l3016_301659


namespace NUMINAMATH_CALUDE_chopped_cube_height_l3016_301650

-- Define the cube
def cube_edge_length : ℝ := 2

-- Define the cut face as an equilateral triangle
def cut_face_is_equilateral_triangle : Prop := sorry

-- Define the remaining height
def remaining_height : ℝ := cube_edge_length - 1

-- Theorem statement
theorem chopped_cube_height :
  cut_face_is_equilateral_triangle →
  remaining_height = 1 := by sorry

end NUMINAMATH_CALUDE_chopped_cube_height_l3016_301650


namespace NUMINAMATH_CALUDE_ruby_height_l3016_301624

/-- Given the heights of various people, prove Ruby's height -/
theorem ruby_height
  (janet_height : ℕ)
  (charlene_height : ℕ)
  (pablo_height : ℕ)
  (ruby_height : ℕ)
  (h1 : janet_height = 62)
  (h2 : charlene_height = 2 * janet_height)
  (h3 : pablo_height = charlene_height + 70)
  (h4 : ruby_height = pablo_height - 2)
  : ruby_height = 192 := by
  sorry


end NUMINAMATH_CALUDE_ruby_height_l3016_301624


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l3016_301693

theorem root_difference_implies_k_value (k : ℝ) :
  (∃ r s : ℝ, r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0) →
  (∃ r s : ℝ, r^2 - k*r + 10 = 0 ∧ s^2 - k*s + 10 = 0) →
  (∀ r s : ℝ, r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0 →
              (r+3)^2 - k*(r+3) + 10 = 0 ∧ (s+3)^2 - k*(s+3) + 10 = 0) →
  k = 3 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l3016_301693


namespace NUMINAMATH_CALUDE_vector_b_proof_l3016_301619

def vector_a : Fin 2 → ℝ := ![2, -1]

theorem vector_b_proof (b : Fin 2 → ℝ) 
  (collinear : ∃ k : ℝ, k > 0 ∧ b = k • vector_a)
  (magnitude : Real.sqrt ((b 0) ^ 2 + (b 1) ^ 2) = 2 * Real.sqrt 5) :
  b = ![4, -2] := by
  sorry

end NUMINAMATH_CALUDE_vector_b_proof_l3016_301619


namespace NUMINAMATH_CALUDE_expression_equality_l3016_301604

theorem expression_equality (x : ℝ) (hx : x > 0) :
  (∃! n : ℕ, n = (if 2 * x^(x+1) = x^(x+1) + x^(x+1) then 1 else 0) +
              (if x^(2*x+2) = x^(x+1) + x^(x+1) then 1 else 0) +
              (if (3*x)^x = x^(x+1) + x^(x+1) then 1 else 0) +
              (if (3*x)^(x+1) = x^(x+1) + x^(x+1) then 1 else 0)) ∧
  n = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equality_l3016_301604


namespace NUMINAMATH_CALUDE_perpendicular_condition_l3016_301613

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp_plane : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Theorem statement
theorem perpendicular_condition 
  (α β : Plane) (l : Line) 
  (h_subset : subset l α) :
  (∀ β, perp_line_plane l β → perp_plane α β) ∧ 
  (∃ β, perp_plane α β ∧ ¬perp_line_plane l β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l3016_301613


namespace NUMINAMATH_CALUDE_forty_two_divisible_by_seven_l3016_301620

theorem forty_two_divisible_by_seven : ∃ k : ℤ, 42 = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_forty_two_divisible_by_seven_l3016_301620


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l3016_301614

/-- Given an initial angle of 60 degrees and a clockwise rotation of 300 degrees,
    the resulting positive acute angle is 120 degrees. -/
theorem rotated_angle_measure (initial_angle rotation_angle : ℝ) : 
  initial_angle = 60 →
  rotation_angle = 300 →
  (360 - (rotation_angle - initial_angle)) % 360 = 120 :=
by sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l3016_301614


namespace NUMINAMATH_CALUDE_first_solution_carbonated_water_percentage_l3016_301610

/-- Represents a solution with lemonade and carbonated water -/
structure Solution where
  lemonade : ℝ
  carbonated_water : ℝ
  sum_to_100 : lemonade + carbonated_water = 100

/-- Proves that the first solution is 80% carbonated water given the conditions -/
theorem first_solution_carbonated_water_percentage
  (solution1 : Solution)
  (solution2 : Solution)
  (h1 : solution1.lemonade = 20)
  (h2 : solution2.lemonade = 45)
  (h3 : solution2.carbonated_water = 55)
  (h_mixture : 0.5 * solution1.carbonated_water + 0.5 * solution2.carbonated_water = 67.5) :
  solution1.carbonated_water = 80 := by
  sorry

#check first_solution_carbonated_water_percentage

end NUMINAMATH_CALUDE_first_solution_carbonated_water_percentage_l3016_301610


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3016_301607

theorem polynomial_factorization (x y : ℝ) : 
  4 * x^2 - 4 * x - y^2 + 4 * y - 3 = (2 * x + y - 3) * (2 * x - y + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3016_301607


namespace NUMINAMATH_CALUDE_lilith_cap_collection_years_l3016_301634

/-- Represents the cap collection problem for Lilith --/
def cap_collection_problem (years : ℕ) : Prop :=
  let first_year_caps := 3 * 12
  let subsequent_year_caps := 5 * 12
  let christmas_caps := 40
  let lost_caps := 15
  let total_caps := 401
  
  first_year_caps +
  (years - 1) * subsequent_year_caps +
  years * christmas_caps -
  years * lost_caps = total_caps

/-- Theorem stating that Lilith has been collecting caps for 5 years --/
theorem lilith_cap_collection_years : 
  ∃ (years : ℕ), years > 0 ∧ cap_collection_problem years ∧ years = 5 := by
  sorry

end NUMINAMATH_CALUDE_lilith_cap_collection_years_l3016_301634


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3016_301696

theorem rectangle_perimeter (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let area := l * b
  area = 363 → 2 * (l + b) = 88 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3016_301696


namespace NUMINAMATH_CALUDE_primes_between_30_and_50_l3016_301690

/-- Count of prime numbers in a given range -/
def countPrimes (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).filter (fun i => Nat.Prime (i + a)) |>.card

/-- The theorem stating that there are 5 prime numbers between 30 and 50 -/
theorem primes_between_30_and_50 : countPrimes 31 49 = 5 := by
  sorry

end NUMINAMATH_CALUDE_primes_between_30_and_50_l3016_301690


namespace NUMINAMATH_CALUDE_box_surface_area_proof_l3016_301685

noncomputable def surface_area_of_box (a b c : ℝ) : ℝ :=
  2 * (a * b + b * c + c * a)

theorem box_surface_area_proof (a b c : ℝ) 
  (h1 : a ≤ b) 
  (h2 : b ≤ c) 
  (h3 : c = 2 * a) 
  (h4 : 4 * a + 4 * b + 4 * c = 180) 
  (h5 : Real.sqrt (a^2 + b^2 + c^2) = 25) :
  ∃ ε > 0, |surface_area_of_box a b c - 1051.540| < ε :=
sorry

end NUMINAMATH_CALUDE_box_surface_area_proof_l3016_301685


namespace NUMINAMATH_CALUDE_percentage_calculation_l3016_301656

theorem percentage_calculation (P : ℝ) : 
  P * (0.3 * (0.5 * 4000)) = 90 → P = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3016_301656


namespace NUMINAMATH_CALUDE_october_visitors_l3016_301672

theorem october_visitors (oct nov dec : ℕ) : 
  nov = oct * 115 / 100 →
  dec = nov + 15 →
  oct + nov + dec = 345 →
  oct = 100 := by
sorry

end NUMINAMATH_CALUDE_october_visitors_l3016_301672


namespace NUMINAMATH_CALUDE_potato_peeling_result_l3016_301651

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ := 60
  homer_rate : ℕ := 4
  christen_rate : ℕ := 6
  homer_solo_time : ℕ := 5

/-- Calculates the number of potatoes Christen peeled and the total time taken -/
def peel_potatoes (scenario : PotatoPeeling) : ℕ × ℕ := by
  sorry

/-- Theorem stating the correct result of the potato peeling scenario -/
theorem potato_peeling_result (scenario : PotatoPeeling) :
  peel_potatoes scenario = (24, 9) := by
  sorry

end NUMINAMATH_CALUDE_potato_peeling_result_l3016_301651


namespace NUMINAMATH_CALUDE_grocer_sales_l3016_301608

theorem grocer_sales (sales : List ℕ) (average : ℕ) : 
  sales = [800, 900, 1000, 800, 900] →
  average = 850 →
  (sales.sum + 700) / 6 = average →
  700 = 6 * average - sales.sum :=
by sorry

end NUMINAMATH_CALUDE_grocer_sales_l3016_301608


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3016_301646

-- Define the eccentricities
variable (e₁ e₂ : ℝ)

-- Define the parameters of the hyperbola
variable (a b : ℝ)

-- Define the coordinates of the intersection point M
variable (x y : ℝ)

-- Define the coordinates of the foci
variable (c : ℝ)

-- Theorem statement
theorem hyperbola_eccentricity_range 
  (h1 : a > b) 
  (h2 : b > 0)
  (h3 : x^2 / a^2 - y^2 / b^2 = 1)  -- Hyperbola equation
  (h4 : x > 0 ∧ y > 0)  -- M is in the first quadrant
  (h5 : (x + c) * (x - c) + y^2 = 0)  -- F₁M · F₂M = 0
  (h6 : 3/4 ≤ e₁ ∧ e₁ ≤ 3*Real.sqrt 10/10)  -- Range of e₁
  (h7 : 1/e₁^2 + 1/e₂^2 = 1)  -- Relationship between e₁ and e₂
  : 3*Real.sqrt 2/4 ≤ e₂ ∧ e₂ < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3016_301646


namespace NUMINAMATH_CALUDE_probability_rain_given_east_wind_l3016_301677

/-- The probability of an east wind blowing -/
def P_east_wind : ℚ := 3/10

/-- The probability of rain -/
def P_rain : ℚ := 11/30

/-- The probability of both an east wind blowing and rain -/
def P_east_wind_and_rain : ℚ := 4/15

/-- The probability of rain given that there is an east wind blowing -/
def P_rain_given_east_wind : ℚ := P_east_wind_and_rain / P_east_wind

theorem probability_rain_given_east_wind :
  P_rain_given_east_wind = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_rain_given_east_wind_l3016_301677


namespace NUMINAMATH_CALUDE_class_average_age_problem_l3016_301633

theorem class_average_age_problem (original_students : ℕ) (new_students : ℕ) (new_average_age : ℕ) (average_decrease : ℕ) :
  original_students = 18 →
  new_students = 18 →
  new_average_age = 32 →
  average_decrease = 4 →
  ∃ (original_average : ℕ),
    (original_students * original_average + new_students * new_average_age) / (original_students + new_students) = original_average - average_decrease ∧
    original_average = 40 := by
  sorry

end NUMINAMATH_CALUDE_class_average_age_problem_l3016_301633


namespace NUMINAMATH_CALUDE_inscribed_angles_sum_l3016_301641

/-- Given a circle divided into 15 equal arcs, this theorem proves that
    the sum of two inscribed angles, one subtended by 3 arcs and the other by 5 arcs,
    is equal to 96 degrees. -/
theorem inscribed_angles_sum (circle : Real) (x y : Real) : 
  (circle = 360) →
  (x = 3 * (circle / 15) / 2) →
  (y = 5 * (circle / 15) / 2) →
  x + y = 96 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_angles_sum_l3016_301641


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3016_301628

def i : ℂ := Complex.I

theorem complex_equation_solution :
  ∀ z : ℂ, (2 - i) * z = i^2021 → z = -1/5 + 2/5*i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3016_301628


namespace NUMINAMATH_CALUDE_game_download_time_l3016_301670

theorem game_download_time (total_size : ℕ) (downloaded : ℕ) (speed : ℕ) : 
  total_size = 880 → downloaded = 310 → speed = 3 → 
  (total_size - downloaded) / speed = 190 := by
  sorry

end NUMINAMATH_CALUDE_game_download_time_l3016_301670


namespace NUMINAMATH_CALUDE_carolyns_project_time_l3016_301635

/-- Represents the embroidering project with given parameters -/
structure EmbroideringProject where
  stitches_per_minute : ℕ
  flower_stitches : ℕ
  unicorn_stitches : ℕ
  godzilla_stitches : ℕ
  num_flowers : ℕ
  num_unicorns : ℕ
  num_godzillas : ℕ
  embroidering_time_before_break : ℕ
  break_duration : ℕ

/-- Calculates the total time needed for the embroidering project -/
def total_time (project : EmbroideringProject) : ℕ :=
  sorry

/-- Theorem stating that the total time for Carolyn's project is 1265 minutes -/
theorem carolyns_project_time :
  let project : EmbroideringProject := {
    stitches_per_minute := 4,
    flower_stitches := 60,
    unicorn_stitches := 180,
    godzilla_stitches := 800,
    num_flowers := 50,
    num_unicorns := 3,
    num_godzillas := 1,
    embroidering_time_before_break := 30,
    break_duration := 5
  }
  total_time project = 1265 := by sorry

end NUMINAMATH_CALUDE_carolyns_project_time_l3016_301635


namespace NUMINAMATH_CALUDE_range_of_expression_l3016_301699

theorem range_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 ≤ x^2 + y^2 + Real.sqrt (x * y) ∧ x^2 + y^2 + Real.sqrt (x * y) ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l3016_301699


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3016_301605

theorem square_area_from_diagonal (diagonal : Real) (area : Real) :
  diagonal = 10 → area = diagonal^2 / 2 → area = 50 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3016_301605


namespace NUMINAMATH_CALUDE_stream_speed_l3016_301695

/-- Proves that the speed of a stream is 3 kmph given specific boat travel times -/
theorem stream_speed (boat_speed : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) : 
  boat_speed = 15 →
  downstream_time = 1 →
  upstream_time = 1.5 →
  (boat_speed + 3) * downstream_time = (boat_speed - 3) * upstream_time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l3016_301695


namespace NUMINAMATH_CALUDE_M_in_fourth_quadrant_l3016_301675

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point M -/
def M : Point :=
  { x := 3, y := -2 }

/-- Theorem stating that M is in the fourth quadrant -/
theorem M_in_fourth_quadrant : is_in_fourth_quadrant M := by
  sorry


end NUMINAMATH_CALUDE_M_in_fourth_quadrant_l3016_301675


namespace NUMINAMATH_CALUDE_sum_of_squares_l3016_301652

theorem sum_of_squares (x y z : ℤ) : 
  x + y + 57 = 0 → y - z + 17 = 0 → x - z + 44 = 0 → x^2 + y^2 + z^2 = 1993 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3016_301652


namespace NUMINAMATH_CALUDE_fraction_simplification_l3016_301662

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) :
  (2*x - 1) / (x - 1) + x / (1 - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3016_301662


namespace NUMINAMATH_CALUDE_minimum_employees_l3016_301639

theorem minimum_employees (work_days : ℕ) (rest_days : ℕ) (daily_requirement : ℕ) : 
  work_days = 5 →
  rest_days = 2 →
  daily_requirement = 32 →
  ∃ min_employees : ℕ,
    min_employees = (daily_requirement * 7 + work_days - 1) / work_days ∧
    min_employees * work_days ≥ daily_requirement * 7 ∧
    ∀ n : ℕ, n < min_employees → n * work_days < daily_requirement * 7 :=
by
  sorry

#eval (32 * 7 + 5 - 1) / 5  -- Should output 45

end NUMINAMATH_CALUDE_minimum_employees_l3016_301639


namespace NUMINAMATH_CALUDE_union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l3016_301623

variable {U : Type} -- Universe set
variable (A B C : Set U) -- Sets A, B, C in the universe U

-- Commutativity
theorem union_comm : A ∪ B = B ∪ A := by sorry
theorem inter_comm : A ∩ B = B ∩ A := by sorry

-- Associativity
theorem union_assoc : A ∪ (B ∪ C) = (A ∪ B) ∪ C := by sorry
theorem inter_assoc : A ∩ (B ∩ C) = (A ∩ B) ∩ C := by sorry

-- Distributivity
theorem inter_union_distrib : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := by sorry
theorem union_inter_distrib : A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) := by sorry

-- Idempotence
theorem union_idem : A ∪ A = A := by sorry
theorem inter_idem : A ∩ A = A := by sorry

-- De Morgan's Laws
theorem de_morgan_union : (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ := by sorry
theorem de_morgan_inter : (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ := by sorry

end NUMINAMATH_CALUDE_union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l3016_301623


namespace NUMINAMATH_CALUDE_no_solution_implies_m_geq_two_l3016_301637

theorem no_solution_implies_m_geq_two (m : ℝ) :
  (∀ x : ℝ, ¬(2*x - 1 < 3 ∧ x > m)) → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_geq_two_l3016_301637


namespace NUMINAMATH_CALUDE_total_tickets_sold_l3016_301661

theorem total_tickets_sold (adult_price child_price : ℕ) 
  (adult_tickets child_tickets total_receipts : ℕ) :
  adult_price = 12 →
  child_price = 4 →
  adult_tickets = 90 →
  child_tickets = 40 →
  total_receipts = 840 →
  adult_tickets * adult_price + child_tickets * child_price = total_receipts →
  adult_tickets + child_tickets = 130 := by
sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l3016_301661


namespace NUMINAMATH_CALUDE_max_sum_given_sum_squares_and_product_l3016_301664

theorem max_sum_given_sum_squares_and_product (x y : ℝ) 
  (h1 : x^2 + y^2 = 130) (h2 : x * y = 45) : 
  x + y ≤ Real.sqrt 220 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_sum_squares_and_product_l3016_301664


namespace NUMINAMATH_CALUDE_cube_root_existence_l3016_301602

theorem cube_root_existence : ∀ y : ℝ, ∃ x : ℝ, x^3 = y := by
  sorry

end NUMINAMATH_CALUDE_cube_root_existence_l3016_301602


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l3016_301606

theorem min_value_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1/x + 4/y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 1/x₀ + 4/y₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l3016_301606


namespace NUMINAMATH_CALUDE_weight_problem_l3016_301669

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions and the average weight of b and c is 41 kg. -/
theorem weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  b = 27 →
  (b + c) / 2 = 41 := by
sorry

end NUMINAMATH_CALUDE_weight_problem_l3016_301669


namespace NUMINAMATH_CALUDE_tile_1x1_position_l3016_301686

/-- Represents a position in the 7x7 grid -/
structure Position where
  row : Fin 7
  col : Fin 7

/-- Represents a 1x3 tile -/
structure Tile1x3 where
  start : Position
  horizontal : Bool

/-- Represents the placement of tiles in the 7x7 grid -/
structure TilePlacement where
  tiles1x3 : Finset Tile1x3
  tile1x1 : Position

/-- Predicate to check if a position is in the center or adjacent to the edges -/
def isCenterOrEdgeAdjacent (p : Position) : Prop :=
  (p.row = 3 ∧ p.col = 3) ∨ 
  p.row = 0 ∨ p.row = 6 ∨ p.col = 0 ∨ p.col = 6 ∨
  p.row = 1 ∨ p.row = 5 ∨ p.col = 1 ∨ p.col = 5

/-- Main theorem: The 1x1 tile must be in the center or adjacent to the edges -/
theorem tile_1x1_position (placement : TilePlacement) 
  (h1 : placement.tiles1x3.card = 16) 
  (h2 : ∀ t ∈ placement.tiles1x3, t.start.row < 7 ∧ t.start.col < 7) 
  (h3 : ∀ t ∈ placement.tiles1x3, 
    if t.horizontal 
    then t.start.col < 5 
    else t.start.row < 5) :
  isCenterOrEdgeAdjacent placement.tile1x1 :=
sorry

end NUMINAMATH_CALUDE_tile_1x1_position_l3016_301686


namespace NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l3016_301663

/-- Simple interest rate for a sum that doubles in 10 years -/
theorem simple_interest_rate_for_doubling (principal : ℝ) (h : principal > 0) :
  let years : ℝ := 10
  let final_amount : ℝ := 2 * principal
  let rate : ℝ := (final_amount - principal) / (principal * years) * 100
  rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l3016_301663


namespace NUMINAMATH_CALUDE_max_positive_cyclic_sequence_l3016_301680

theorem max_positive_cyclic_sequence (x : Fin 2022 → ℝ) 
  (h_nonzero : ∀ i, x i ≠ 0)
  (h_inequality : ∀ i : Fin 2022, x i + 1 / x (Fin.succ i) < 0)
  (h_cyclic : x 0 = x (Fin.last 2021)) : 
  (Finset.filter (fun i => x i > 0) Finset.univ).card ≤ 1010 := by
  sorry

end NUMINAMATH_CALUDE_max_positive_cyclic_sequence_l3016_301680


namespace NUMINAMATH_CALUDE_negative_one_to_2002_is_smallest_positive_integer_l3016_301665

theorem negative_one_to_2002_is_smallest_positive_integer :
  (-1 : ℤ) ^ 2002 = 1 ∧ ∀ n : ℤ, n > 0 → n ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_negative_one_to_2002_is_smallest_positive_integer_l3016_301665


namespace NUMINAMATH_CALUDE_min_distance_between_graphs_l3016_301684

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the distance function between the two graphs
def distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem min_distance_between_graphs :
  ∃ (min_dist : ℝ), min_dist = 3/4 ∧ ∀ (x : ℝ), distance x ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_graphs_l3016_301684


namespace NUMINAMATH_CALUDE_number_problem_l3016_301689

theorem number_problem (x : ℚ) : x + (-5/12) - (-5/2) = 1/3 → x = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3016_301689


namespace NUMINAMATH_CALUDE_subset_condition_l3016_301666

theorem subset_condition (a b : ℝ) : 
  let A : Set ℝ := {x | x^2 - 1 = 0}
  let B : Set ℝ := {y | y^2 - 2*a*y + b = 0}
  (B ⊆ A) ∧ (B ≠ ∅) → 
  ((a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) := by
sorry

end NUMINAMATH_CALUDE_subset_condition_l3016_301666


namespace NUMINAMATH_CALUDE_systematic_sample_interval_count_l3016_301682

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  totalPopulation : ℕ
  sampleSize : ℕ
  intervalStart : ℕ
  intervalEnd : ℕ

/-- Calculates the number of selected items within a given interval in a systematic sample -/
def selectedInInterval (s : SystematicSample) : ℕ :=
  let stepSize := s.totalPopulation / s.sampleSize
  let intervalSize := s.intervalEnd - s.intervalStart + 1
  intervalSize / stepSize

/-- The main theorem statement -/
theorem systematic_sample_interval_count :
  let s : SystematicSample := {
    totalPopulation := 840,
    sampleSize := 21,
    intervalStart := 481,
    intervalEnd := 720
  }
  selectedInInterval s = 6 := by sorry

end NUMINAMATH_CALUDE_systematic_sample_interval_count_l3016_301682


namespace NUMINAMATH_CALUDE_fraction_product_equivalence_l3016_301642

theorem fraction_product_equivalence :
  ∀ x : ℝ, x ≠ 1 → ((x + 2) / (x - 1) ≥ 0 ↔ (x + 2) * (x - 1) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_equivalence_l3016_301642


namespace NUMINAMATH_CALUDE_point_on_y_axis_l3016_301600

/-- A point M with coordinates (m+3, m+1) lies on the y-axis if and only if its coordinates are (0, -2) -/
theorem point_on_y_axis (m : ℝ) : 
  (m + 3 = 0 ∧ m + 1 = -2) ↔ (m + 3 = 0 ∧ m + 1 = -2) :=
by sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l3016_301600


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3016_301671

theorem simplify_trig_expression (x : ℝ) : 
  (3 + 3 * Real.sin x - 3 * Real.cos x) / (3 + 3 * Real.sin x + 3 * Real.cos x) = Real.tan (x / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3016_301671


namespace NUMINAMATH_CALUDE_new_years_eve_appetizer_cost_l3016_301626

def cost_per_person (chips_cost creme_fraiche_cost caviar_cost : ℚ) (num_people : ℕ) : ℚ :=
  (chips_cost + creme_fraiche_cost + caviar_cost) / num_people

theorem new_years_eve_appetizer_cost :
  cost_per_person 3 5 73 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_new_years_eve_appetizer_cost_l3016_301626


namespace NUMINAMATH_CALUDE_mark_to_jenna_ratio_l3016_301658

/-- The number of math problems in the homework -/
def total_problems : ℕ := 20

/-- The number of problems Martha finished -/
def martha_problems : ℕ := 2

/-- The number of problems Jenna finished -/
def jenna_problems : ℕ := 4 * martha_problems - 2

/-- The number of problems Angela finished -/
def angela_problems : ℕ := 9

/-- The number of problems Mark finished -/
def mark_problems : ℕ := total_problems - (martha_problems + jenna_problems + angela_problems)

/-- Theorem stating the ratio of problems Mark finished to problems Jenna finished -/
theorem mark_to_jenna_ratio : 
  (mark_problems : ℚ) / jenna_problems = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_mark_to_jenna_ratio_l3016_301658


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l3016_301649

theorem consecutive_negative_integers_sum (x : ℤ) : 
  x < 0 ∧ x * (x + 1) = 3080 → x + (x + 1) = -111 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l3016_301649


namespace NUMINAMATH_CALUDE_fraction_problem_l3016_301697

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/4) * F * (2/5) * N = 35)
  (h2 : (40/100) * N = 420) : F = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3016_301697


namespace NUMINAMATH_CALUDE_smallest_twin_prime_pair_mean_l3016_301683

/-- Twin prime pair -/
def is_twin_prime_pair (p q : Nat) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ q = p + 2

/-- Smallest twin prime pair -/
def smallest_twin_prime_pair (p q : Nat) : Prop :=
  is_twin_prime_pair p q ∧ ∀ (r s : Nat), is_twin_prime_pair r s → p ≤ r

/-- Arithmetic mean of two numbers -/
def arithmetic_mean (a b : Nat) : Rat :=
  (a + b : Rat) / 2

theorem smallest_twin_prime_pair_mean :
  ∃ (p q : Nat), smallest_twin_prime_pair p q ∧ arithmetic_mean p q = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_twin_prime_pair_mean_l3016_301683


namespace NUMINAMATH_CALUDE_paint_replacement_theorem_l3016_301667

def paint_replacement_fractions (initial_red initial_blue initial_green : ℚ)
                                (replacement_red replacement_blue replacement_green : ℚ)
                                (final_red final_blue final_green : ℚ) : Prop :=
  let r := (initial_red - final_red) / (initial_red - replacement_red)
  let b := (initial_blue - final_blue) / (initial_blue - replacement_blue)
  let g := (initial_green - final_green) / (initial_green - replacement_green)
  r = 2/3 ∧ b = 3/5 ∧ g = 7/15

theorem paint_replacement_theorem :
  paint_replacement_fractions (60/100) (40/100) (25/100) (30/100) (15/100) (10/100) (40/100) (25/100) (18/100) :=
by
  sorry

end NUMINAMATH_CALUDE_paint_replacement_theorem_l3016_301667


namespace NUMINAMATH_CALUDE_max_geometric_mean_of_sequence_l3016_301654

/-- Given a sequence of six numbers where one number is 1, any three consecutive numbers have the same arithmetic mean, and the arithmetic mean of all six numbers is A, the maximum value of the geometric mean of any three consecutive numbers is ∛((3A - 1)² / 4). -/
theorem max_geometric_mean_of_sequence (A : ℝ) (seq : Fin 6 → ℝ) 
  (h1 : ∃ i, seq i = 1)
  (h2 : ∀ i : Fin 4, (seq i + seq (i + 1) + seq (i + 2)) / 3 = 
                     (seq (i + 1) + seq (i + 2) + seq (i + 3)) / 3)
  (h3 : (seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5) / 6 = A) :
  ∃ i : Fin 4, (seq i * seq (i + 1) * seq (i + 2))^(1/3 : ℝ) ≤ ((3*A - 1)^2 / 4)^(1/3 : ℝ) ∧ 
  ∀ j : Fin 4, (seq j * seq (j + 1) * seq (j + 2))^(1/3 : ℝ) ≤ 
               (seq i * seq (i + 1) * seq (i + 2))^(1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_geometric_mean_of_sequence_l3016_301654


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3016_301603

theorem quadratic_equations_solutions :
  (∀ x : ℝ, 3 * x^2 - 6 * x - 2 = 0 ↔ x = 1 + Real.sqrt 15 / 3 ∨ x = 1 - Real.sqrt 15 / 3) ∧
  (∀ x : ℝ, x^2 - 2 - 3 * x = 0 ↔ x = (3 + Real.sqrt 17) / 2 ∨ x = (3 - Real.sqrt 17) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3016_301603


namespace NUMINAMATH_CALUDE_cone_base_area_l3016_301617

/-- The area of the base of a cone with slant height 10 and lateral surface that unfolds into a semicircle -/
theorem cone_base_area (l : ℝ) (r : ℝ) : 
  l = 10 →                       -- Slant height is 10
  l = 2 * r →                    -- Lateral surface unfolds into a semicircle
  π * r^2 = 25 * π :=            -- Area of the base is 25π
by sorry

end NUMINAMATH_CALUDE_cone_base_area_l3016_301617


namespace NUMINAMATH_CALUDE_min_distance_vectors_l3016_301694

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem min_distance_vectors (a b : ℝ × ℝ) 
  (h1 : angle_between_vectors a b = 2 * Real.pi / 3)
  (h2 : a.1 * b.1 + a.2 * b.2 = -1) : 
  ∀ (c d : ℝ × ℝ), angle_between_vectors c d = 2 * Real.pi / 3 → 
  c.1 * d.1 + c.2 * d.2 = -1 → 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≤ Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_vectors_l3016_301694


namespace NUMINAMATH_CALUDE_horse_race_probability_l3016_301643

theorem horse_race_probability (X Y Z : ℝ) 
  (no_draw : X + Y + Z = 1)
  (prob_X : X = 1/4)
  (prob_Y : Y = 3/5) : 
  Z = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_horse_race_probability_l3016_301643


namespace NUMINAMATH_CALUDE_irrational_free_iff_zero_and_rational_l3016_301627

def M (a b c d : ℝ) : Set ℝ :=
  {y | ∃ x, y = a * x^3 + b * x^2 + c * x + d}

theorem irrational_free_iff_zero_and_rational (a b c d : ℝ) :
  (∀ y ∈ M a b c d, ∃ (q : ℚ), (y : ℝ) = q) ↔
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ ∃ (q : ℚ), (d : ℝ) = q) :=
sorry

end NUMINAMATH_CALUDE_irrational_free_iff_zero_and_rational_l3016_301627


namespace NUMINAMATH_CALUDE_pool_cleaning_threshold_l3016_301638

/-- Represents the pool maintenance scenario -/
structure PoolMaintenance where
  capacity : ℕ  -- Pool capacity in milliliters
  splash_per_jump : ℕ  -- Amount of water splashed out per jump in milliliters
  num_jumps : ℕ  -- Number of jumps before cleaning

/-- Calculates the percentage of water remaining in the pool after jumps -/
def remaining_water_percentage (p : PoolMaintenance) : ℚ :=
  let remaining_water := p.capacity - p.splash_per_jump * p.num_jumps
  (remaining_water : ℚ) / (p.capacity : ℚ) * 100

/-- Theorem stating that the remaining water percentage is 80% for the given scenario -/
theorem pool_cleaning_threshold (p : PoolMaintenance) 
  (h1 : p.capacity = 2000000)
  (h2 : p.splash_per_jump = 400)
  (h3 : p.num_jumps = 1000) :
  remaining_water_percentage p = 80 := by
  sorry


end NUMINAMATH_CALUDE_pool_cleaning_threshold_l3016_301638


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3016_301615

theorem polynomial_remainder (y : ℝ) : 
  ∃ (Q : ℝ → ℝ), y^50 = (y^2 - 5*y + 6) * Q y + ((3^50 - 2^50)*y + (2^50 - 2*3^50 + 2*2^50)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3016_301615


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3016_301647

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3016_301647


namespace NUMINAMATH_CALUDE_tiffany_cans_l3016_301636

theorem tiffany_cans (x : ℕ) : x + 4 = 8 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_cans_l3016_301636


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3016_301698

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 3) ↔ d ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3016_301698


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3016_301660

theorem sufficient_not_necessary_condition :
  (∃ x : ℝ, x > 2 ∧ (x - 1)^2 > 1) ∧
  (∃ x : ℝ, (x - 1)^2 > 1 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3016_301660


namespace NUMINAMATH_CALUDE_vasilya_wins_l3016_301688

/-- Represents a stick with a given length -/
structure Stick where
  length : ℝ
  length_pos : length > 0

/-- Represents a game state with a list of sticks -/
structure GameState where
  sticks : List Stick

/-- Represents a player's strategy for breaking sticks -/
def Strategy := GameState → Nat → Stick

/-- Defines the initial game state with a single 10 cm stick -/
def initialState : GameState :=
  { sticks := [{ length := 10, length_pos := by norm_num }] }

/-- Defines the game play for 18 breaks with alternating players -/
def playGame (petyaStrategy vasilyaStrategy : Strategy) : GameState :=
  sorry -- Implementation of game play

/-- Theorem stating that Vasilya can always ensure at least one stick is not shorter than 1 cm -/
theorem vasilya_wins (petyaStrategy : Strategy) : 
  ∃ (vasilyaStrategy : Strategy), ∃ (s : Stick), s ∈ (playGame petyaStrategy vasilyaStrategy).sticks ∧ s.length ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_vasilya_wins_l3016_301688


namespace NUMINAMATH_CALUDE_min_value_theorem_l3016_301629

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + 2*y + 3*z = 6) : 
  (1/x + 4/y + 9/z) ≥ 98/3 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀ + 2*y₀ + 3*z₀ = 6 ∧ 1/x₀ + 4/y₀ + 9/z₀ = 98/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3016_301629


namespace NUMINAMATH_CALUDE_sum_less_than_sqrt_three_sum_squares_l3016_301618

theorem sum_less_than_sqrt_three_sum_squares (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a + b + c < Real.sqrt (3 * (a^2 + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_sum_less_than_sqrt_three_sum_squares_l3016_301618


namespace NUMINAMATH_CALUDE_expression_simplification_l3016_301631

variable (a b : ℝ)

theorem expression_simplification (h : a ≠ -1) :
  b * (a - 1 + 1 / (a + 1)) / ((a^2 + 2*a) / (a + 1)) = a * b / (a + 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3016_301631


namespace NUMINAMATH_CALUDE_price_difference_chips_pretzels_l3016_301632

/-- The price difference between chips and pretzels -/
theorem price_difference_chips_pretzels :
  ∀ (pretzel_price chip_price : ℕ),
    pretzel_price = 4 →
    2 * chip_price + 2 * pretzel_price = 22 →
    chip_price > pretzel_price →
    chip_price - pretzel_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_chips_pretzels_l3016_301632


namespace NUMINAMATH_CALUDE_expression_value_l3016_301655

theorem expression_value : 1 - (-2) - 3 - (-4) - 5 - (-6) - 7 - (-8) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3016_301655


namespace NUMINAMATH_CALUDE_digits_of_product_l3016_301687

theorem digits_of_product : ∃ n : ℕ, n > 0 ∧ (2^15 * 5^10 * 12 : ℕ) < 10^n ∧ (2^15 * 5^10 * 12 : ℕ) ≥ 10^(n-1) ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_product_l3016_301687


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l3016_301679

theorem cubic_polynomial_root (Q : ℝ → ℝ) : 
  (∀ x, Q x = x^3 - 6*x^2 + 12*x - 11) →
  (∃ a b c : ℤ, ∀ x, Q x = x^3 + a*x^2 + b*x + c) →
  Q (Real.rpow 3 (1/3) + 2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l3016_301679


namespace NUMINAMATH_CALUDE_student_arrangement_l3016_301681

theorem student_arrangement (n m k : ℕ) (hn : n = 5) (hm : m = 4) (hk : k = 2) : 
  (Nat.choose m k / 2) * (Nat.factorial n / Nat.factorial (n - k)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_l3016_301681


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3016_301645

/-- A line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- The polar axis -/
def polarAxis : Set (ℝ × ℝ) :=
  {p | p.2 = 0}

/-- A line is parallel to the polar axis -/
def isParallelToPolarAxis (l : PolarLine) : Prop :=
  ∃ c : ℝ, ∀ ρ θ : ℝ, l.equation ρ θ ↔ ρ * Real.sin θ = c

/-- The theorem stating that a line parallel to the polar axis has the equation ρ sin θ = c -/
theorem parallel_line_equation (l : PolarLine) :
  isParallelToPolarAxis l ↔
  ∃ c : ℝ, ∀ ρ θ : ℝ, l.equation ρ θ ↔ ρ * Real.sin θ = c :=
sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l3016_301645


namespace NUMINAMATH_CALUDE_apples_picked_total_l3016_301622

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- The total number of apples picked -/
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_picked_total : total_apples = 11 := by
  sorry

end NUMINAMATH_CALUDE_apples_picked_total_l3016_301622


namespace NUMINAMATH_CALUDE_parabola_minimum_distance_product_parabola_minimum_distance_product_achieved_l3016_301621

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the product of distances from A and B to F
def distance_product (x1 x2 : ℝ) : ℝ := (x1 + 1) * (x2 + 1)

theorem parabola_minimum_distance_product :
  ∀ k : ℝ, ∀ x1 x2 : ℝ,
  (∃ y1 y2 : ℝ, parabola x1 y1 ∧ parabola x2 y2 ∧ 
   line_through_focus k x1 y1 ∧ line_through_focus k x2 y2) →
  distance_product x1 x2 ≥ 4 :=
sorry

theorem parabola_minimum_distance_product_achieved :
  ∃ k x1 x2 : ℝ, ∃ y1 y2 : ℝ,
  parabola x1 y1 ∧ parabola x2 y2 ∧ 
  line_through_focus k x1 y1 ∧ line_through_focus k x2 y2 ∧
  distance_product x1 x2 = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_minimum_distance_product_parabola_minimum_distance_product_achieved_l3016_301621


namespace NUMINAMATH_CALUDE_angle_x_is_60_l3016_301673

/-- Given a geometric configuration where:
  1. y + 140° forms a straight angle
  2. There's a triangle with angles 40°, 80°, and z°
  3. x is an angle opposite to z
Prove that x = 60° -/
theorem angle_x_is_60 (y z x : ℝ) : 
  y + 140 = 180 →  -- Straight angle property
  40 + 80 + z = 180 →  -- Triangle angle sum property
  x = z →  -- Opposite angles are equal
  x = 60 := by sorry

end NUMINAMATH_CALUDE_angle_x_is_60_l3016_301673


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3016_301692

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a) 
  (h_a3 : a 3 = 5) 
  (h_a5 : a 5 = 3) : 
  a 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3016_301692


namespace NUMINAMATH_CALUDE_solution_verification_l3016_301640

theorem solution_verification (x : ℚ) : 
  x = 22 / 5 ↔ 10 * (5 * x + 4) - 4 = -4 * (2 - 15 * x) := by
  sorry

end NUMINAMATH_CALUDE_solution_verification_l3016_301640


namespace NUMINAMATH_CALUDE_cos_18_degrees_l3016_301644

theorem cos_18_degrees : Real.cos (18 * π / 180) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_18_degrees_l3016_301644


namespace NUMINAMATH_CALUDE_function_inequality_l3016_301676

theorem function_inequality (a x : ℝ) (h1 : a ≥ Real.exp (-2)) (h2 : x > 0) :
  a * x * Real.exp x - (x + 1)^2 ≥ Real.log x - x^2 - x - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3016_301676


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3016_301609

theorem election_winner_percentage : 
  let votes : List ℕ := [1036, 4636, 11628]
  let total_votes := votes.sum
  let winning_votes := votes.maximum?
  let winning_percentage := (winning_votes.getD 0 : ℚ) / total_votes * 100
  winning_percentage = 67.2 := by
sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3016_301609


namespace NUMINAMATH_CALUDE_multiple_with_few_digits_l3016_301691

open Nat

theorem multiple_with_few_digits (k : ℕ) (h : k > 1) :
  ∃ p : ℕ, p.gcd k = k ∧ p < k^4 ∧ (∃ (d₁ d₂ d₃ d₄ : ℕ) (h : d₁ < 10 ∧ d₂ < 10 ∧ d₃ < 10 ∧ d₄ < 10),
    ∀ d : ℕ, d ∈ p.digits 10 → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄) :=
by sorry

end NUMINAMATH_CALUDE_multiple_with_few_digits_l3016_301691


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l3016_301616

theorem yellow_marbles_count (total : ℕ) (white yellow green red : ℕ) : 
  total = 50 →
  white = total / 2 →
  green = yellow / 2 →
  red = 7 →
  total = white + yellow + green + red →
  yellow = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l3016_301616


namespace NUMINAMATH_CALUDE_no_real_solutions_l3016_301653

theorem no_real_solutions :
  ¬∃ (x : ℝ), (8 * x^2 + 150 * x - 5) / (3 * x + 50) = 4 * x + 7 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3016_301653


namespace NUMINAMATH_CALUDE_gp_common_ratio_l3016_301611

/-- 
Theorem: In a geometric progression where the ratio of the sum of the first 6 terms 
to the sum of the first 3 terms is 217, the common ratio is 6.
-/
theorem gp_common_ratio (a r : ℝ) (hr : r ≠ 1) :
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 217 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_gp_common_ratio_l3016_301611


namespace NUMINAMATH_CALUDE_existence_of_nth_root_l3016_301678

theorem existence_of_nth_root (n b : ℕ) (h_n : n > 1) (h_b : b > 1)
  (h : ∀ k : ℕ, k > 1 → ∃ a_k : ℤ, (k : ℤ) ∣ (b - a_k ^ n)) :
  ∃ A : ℤ, (A : ℤ) ^ n = b :=
sorry

end NUMINAMATH_CALUDE_existence_of_nth_root_l3016_301678


namespace NUMINAMATH_CALUDE_power_of_two_equation_l3016_301612

theorem power_of_two_equation (k : ℤ) : 
  2^2000 - 2^1999 - 2^1998 + 2^1997 = k * 2^1997 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l3016_301612


namespace NUMINAMATH_CALUDE_probability_no_adjacent_same_l3016_301625

/-- The number of people sitting around the circular table -/
def n : ℕ := 5

/-- The number of sides on the die -/
def sides : ℕ := 6

/-- The probability that no two adjacent people roll the same number -/
def prob_no_adjacent_same : ℚ := 25 / 108

/-- Theorem stating the probability of no two adjacent people rolling the same number -/
theorem probability_no_adjacent_same :
  prob_no_adjacent_same = 25 / 108 := by sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_same_l3016_301625
