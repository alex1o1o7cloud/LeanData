import Mathlib

namespace NUMINAMATH_CALUDE_power_product_specific_calculation_l1533_153389

-- Define the power function for rational numbers
def rat_pow (a : ℚ) (n : ℕ) : ℚ := a ^ n

-- Theorem 1: For any rational numbers a and b, and positive integer n, (ab)^n = a^n * b^n
theorem power_product (a b : ℚ) (n : ℕ+) : rat_pow (a * b) n = rat_pow a n * rat_pow b n := by
  sorry

-- Theorem 2: (3/2)^2019 * (-2/3)^2019 = -1
theorem specific_calculation : rat_pow (3/2) 2019 * rat_pow (-2/3) 2019 = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_product_specific_calculation_l1533_153389


namespace NUMINAMATH_CALUDE_square_boundary_product_l1533_153334

theorem square_boundary_product : 
  ∀ (b₁ b₂ : ℝ),
  (∀ x y : ℝ, (y = 3 ∨ y = 7 ∨ x = -1 ∨ x = b₁) → 
    (y = 3 ∨ y = 7 ∨ x = -1 ∨ x = b₂) → 
    (0 ≤ x ∧ x ≤ 4 ∧ 3 ≤ y ∧ y ≤ 7)) →
  (b₁ * b₂ = -15) :=
by sorry

end NUMINAMATH_CALUDE_square_boundary_product_l1533_153334


namespace NUMINAMATH_CALUDE_probability_of_28_l1533_153379

/-- Represents a die with a specific face configuration -/
structure Die :=
  (faces : List ℕ)
  (blank_faces : ℕ)

/-- The first die configuration -/
def die1 : Die :=
  { faces := List.range 18, blank_faces := 1 }

/-- The second die configuration -/
def die2 : Die :=
  { faces := (List.range 7) ++ (List.range' 9 20), blank_faces := 1 }

/-- Calculates the probability of a specific sum when rolling two dice -/
def probability_of_sum (d1 d2 : Die) (target_sum : ℕ) : ℚ :=
  sorry

theorem probability_of_28 :
  probability_of_sum die1 die2 28 = 1 / 40 := by sorry

end NUMINAMATH_CALUDE_probability_of_28_l1533_153379


namespace NUMINAMATH_CALUDE_mass_of_man_l1533_153351

/-- The mass of a man who causes a boat to sink by a certain amount in water. -/
theorem mass_of_man (boat_length boat_breadth boat_sink_depth water_density : ℝ) 
  (h1 : boat_length = 3)
  (h2 : boat_breadth = 2)
  (h3 : boat_sink_depth = 0.02)
  (h4 : water_density = 1000) : 
  boat_length * boat_breadth * boat_sink_depth * water_density = 120 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_man_l1533_153351


namespace NUMINAMATH_CALUDE_proportion_solution_l1533_153312

theorem proportion_solution : 
  ∀ x : ℚ, (2 : ℚ) / 5 = (4 : ℚ) / 3 / x → x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l1533_153312


namespace NUMINAMATH_CALUDE_equality_multiplication_l1533_153317

theorem equality_multiplication (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end NUMINAMATH_CALUDE_equality_multiplication_l1533_153317


namespace NUMINAMATH_CALUDE_bicycle_sale_price_l1533_153304

def price_store_p : ℝ := 200

def regular_price_store_q : ℝ := price_store_p * 1.15

def sale_price_store_q : ℝ := regular_price_store_q * 0.9

theorem bicycle_sale_price : sale_price_store_q = 207 := by sorry

end NUMINAMATH_CALUDE_bicycle_sale_price_l1533_153304


namespace NUMINAMATH_CALUDE_factory_temporary_stats_l1533_153337

/-- Represents the different employee categories in the factory -/
inductive EmployeeCategory
  | Technician
  | SkilledLaborer
  | Manager
  | Administrative

/-- Represents the employment status of an employee -/
inductive EmploymentStatus
  | Permanent
  | Temporary

/-- Structure to hold information about each employee category -/
structure CategoryInfo where
  category : EmployeeCategory
  percentage : Float
  permanentPercentage : Float
  weeklyHours : Nat

def factory : List CategoryInfo := [
  { category := EmployeeCategory.Technician, percentage := 0.4, permanentPercentage := 0.6, weeklyHours := 45 },
  { category := EmployeeCategory.SkilledLaborer, percentage := 0.3, permanentPercentage := 0.5, weeklyHours := 40 },
  { category := EmployeeCategory.Manager, percentage := 0.2, permanentPercentage := 0.8, weeklyHours := 50 },
  { category := EmployeeCategory.Administrative, percentage := 0.1, permanentPercentage := 0.9, weeklyHours := 35 }
]

def totalEmployees : Nat := 100

/-- Calculate the percentage of temporary employees -/
def calculateTemporaryPercentage (factoryInfo : List CategoryInfo) : Float :=
  factoryInfo.foldl (fun acc info => 
    acc + info.percentage * (1 - info.permanentPercentage)) 0

/-- Calculate the total weekly hours worked by temporary employees -/
def calculateTemporaryHours (factoryInfo : List CategoryInfo) (totalEmp : Nat) : Float :=
  factoryInfo.foldl (fun acc info => 
    acc + (info.percentage * totalEmp.toFloat * (1 - info.permanentPercentage) * info.weeklyHours.toFloat)) 0

theorem factory_temporary_stats :
  calculateTemporaryPercentage factory = 0.36 ∧ 
  calculateTemporaryHours factory totalEmployees = 1555 := by
  sorry


end NUMINAMATH_CALUDE_factory_temporary_stats_l1533_153337


namespace NUMINAMATH_CALUDE_toms_average_score_l1533_153366

theorem toms_average_score (subjects_sem1 subjects_sem2 : ℕ)
  (avg_score_sem1 avg_score_5_sem2 avg_score_all : ℚ) :
  subjects_sem1 = 3 →
  subjects_sem2 = 7 →
  avg_score_sem1 = 85 →
  avg_score_5_sem2 = 78 →
  avg_score_all = 80 →
  (subjects_sem1 * avg_score_sem1 + 5 * avg_score_5_sem2 + 2 * ((subjects_sem1 + subjects_sem2) * avg_score_all - subjects_sem1 * avg_score_sem1 - 5 * avg_score_5_sem2) / 2) / (subjects_sem1 + subjects_sem2) = avg_score_all :=
by sorry

end NUMINAMATH_CALUDE_toms_average_score_l1533_153366


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1533_153359

/-- Given a quadratic equation x^2 + 2x + k = 0 with two equal real roots, prove that k = 1 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 + 2*y + k = 0 → y = x) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1533_153359


namespace NUMINAMATH_CALUDE_lcm_of_72_108_126_156_l1533_153371

theorem lcm_of_72_108_126_156 : Nat.lcm 72 (Nat.lcm 108 (Nat.lcm 126 156)) = 19656 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_72_108_126_156_l1533_153371


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l1533_153373

/-- The center of a circle satisfying given conditions -/
theorem circle_center_coordinates :
  ∃ (x y : ℝ),
    (x - 2*y = 0) ∧
    (3*x - 4*y = 20) ∧
    (x = 20 ∧ y = 10) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l1533_153373


namespace NUMINAMATH_CALUDE_package_volume_calculation_l1533_153368

/-- Proves that the total volume needed to package the collection is 3,060,000 cubic inches -/
theorem package_volume_calculation (box_length box_width box_height : ℕ) 
  (cost_per_box total_cost : ℚ) : 
  box_length = 20 →
  box_width = 20 →
  box_height = 15 →
  cost_per_box = 7/10 →
  total_cost = 357 →
  (box_length * box_width * box_height) * (total_cost / cost_per_box) = 3060000 :=
by sorry

end NUMINAMATH_CALUDE_package_volume_calculation_l1533_153368


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l1533_153355

/-- Given two right triangles with sides 3, 4, and 5, where one triangle has a square
    inscribed with a vertex at the right angle (side length x) and the other has a square
    inscribed with a side on the hypotenuse (side length y), prove that x/y = 37/35 -/
theorem inscribed_squares_ratio (x y : ℝ) : 
  (∃ (a b c d : ℝ), 
    a^2 + b^2 = c^2 ∧ a = 3 ∧ b = 4 ∧ c = 5 ∧
    x^2 = a * b - (a - x) * (b - x) ∧
    y * (a + b) = c * y) →
  x / y = 37 / 35 := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l1533_153355


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l1533_153399

theorem quadratic_perfect_square (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 12*x + k = (x + a)^2) ↔ k = 36 :=
sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l1533_153399


namespace NUMINAMATH_CALUDE_intersection_of_M_and_P_l1533_153385

-- Define the sets M and P
def M : Set ℝ := {x | ∃ y, y = Real.log (x - 3) ∧ x > 3}
def P : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- State the theorem
theorem intersection_of_M_and_P : M ∩ P = {x | 3 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_P_l1533_153385


namespace NUMINAMATH_CALUDE_linear_function_property_l1533_153336

theorem linear_function_property (x : ℝ) : ∃ x > -1, -2 * x + 2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l1533_153336


namespace NUMINAMATH_CALUDE_medal_award_scenario_l1533_153364

/-- The number of ways to award medals in a specific race scenario -/
def medal_award_ways (total_sprinters : ℕ) (italian_sprinters : ℕ) : ℕ :=
  let non_italian_sprinters := total_sprinters - italian_sprinters
  italian_sprinters * non_italian_sprinters * (non_italian_sprinters - 1)

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem medal_award_scenario : medal_award_ways 10 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_medal_award_scenario_l1533_153364


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l1533_153320

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 3 ≤ x ∧ x < 8}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x ≤ 6}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | x ≥ a}

theorem set_operations_and_subset :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 6}) ∧
  (A ∪ B = {x | 2 < x ∧ x < 8}) ∧
  (Aᶜ = {x | x < 3 ∨ x ≥ 8}) ∧
  (∀ a : ℝ, A ⊆ C a → a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l1533_153320


namespace NUMINAMATH_CALUDE_function_extrema_l1533_153365

/-- The function f(x) = 1 + 3x - x³ has a minimum value of -1 and a maximum value of 3. -/
theorem function_extrema :
  ∃ (a b : ℝ), (∀ x : ℝ, 1 + 3 * x - x^3 ≥ a) ∧
                (∃ x : ℝ, 1 + 3 * x - x^3 = a) ∧
                (∀ x : ℝ, 1 + 3 * x - x^3 ≤ b) ∧
                (∃ x : ℝ, 1 + 3 * x - x^3 = b) ∧
                a = -1 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_extrema_l1533_153365


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_implies_complement_in_first_quadrant_l1533_153386

/-- If the terminal side of angle α is in the second quadrant, then π - α is in the first quadrant -/
theorem angle_in_second_quadrant_implies_complement_in_first_quadrant (α : Real) : 
  (∃ k : ℤ, 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) → 
  (∃ m : ℤ, 2 * m * π < π - α ∧ π - α < 2 * m * π + π / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_implies_complement_in_first_quadrant_l1533_153386


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1533_153300

def U : Set ℤ := {x | x^2 < 9}
def A : Set ℤ := {-2, 2}

theorem complement_of_A_in_U :
  U \ A = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1533_153300


namespace NUMINAMATH_CALUDE_square_root_of_nine_l1533_153311

theorem square_root_of_nine (x : ℝ) : x^2 = 9 → (x = 3 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l1533_153311


namespace NUMINAMATH_CALUDE_greatest_n_perfect_cube_l1533_153333

def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def productOfSums (n : ℕ) : ℕ := 
  (sumOfSquares n) * (sumOfSquares (2 * n) - sumOfSquares n)

def isPerfectCube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem greatest_n_perfect_cube : 
  ∀ n : ℕ, n ≤ 2050 → 
    (isPerfectCube (productOfSums n) → n ≤ 2016) ∧ 
    (isPerfectCube (productOfSums 2016)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_perfect_cube_l1533_153333


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1533_153321

theorem inequality_and_equality_condition (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum : x + y + z = 12) : 
  (x / y + y / z + z / x + 3 ≥ Real.sqrt x + Real.sqrt y + Real.sqrt z) ∧
  (x / y + y / z + z / x + 3 = Real.sqrt x + Real.sqrt y + Real.sqrt z ↔ x = 4 ∧ y = 4 ∧ z = 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1533_153321


namespace NUMINAMATH_CALUDE_eq1_solution_eq2_solution_eq3_solution_eq4_solution_eq5_solution_l1533_153307

-- Equation 1: 3x^2 - 15 = 0
theorem eq1_solution (x : ℝ) : x = Real.sqrt 5 ∨ x = -Real.sqrt 5 ↔ 3 * x^2 - 15 = 0 := by sorry

-- Equation 2: x^2 - 8x + 15 = 0
theorem eq2_solution (x : ℝ) : x = 3 ∨ x = 5 ↔ x^2 - 8*x + 15 = 0 := by sorry

-- Equation 3: x^2 - 6x + 7 = 0
theorem eq3_solution (x : ℝ) : x = 3 + Real.sqrt 2 ∨ x = 3 - Real.sqrt 2 ↔ x^2 - 6*x + 7 = 0 := by sorry

-- Equation 4: 2x^2 - 6x + 1 = 0
theorem eq4_solution (x : ℝ) : x = (3 + Real.sqrt 7) / 2 ∨ x = (3 - Real.sqrt 7) / 2 ↔ 2*x^2 - 6*x + 1 = 0 := by sorry

-- Equation 5: (2x^2 + 3x)^2 - 4(2x^2 + 3x) - 5 = 0
theorem eq5_solution (x : ℝ) : x = -5/2 ∨ x = 1 ∨ x = -1/2 ∨ x = -1 ↔ (2*x^2 + 3*x)^2 - 4*(2*x^2 + 3*x) - 5 = 0 := by sorry

end NUMINAMATH_CALUDE_eq1_solution_eq2_solution_eq3_solution_eq4_solution_eq5_solution_l1533_153307


namespace NUMINAMATH_CALUDE_axis_of_symmetry_is_correct_l1533_153326

/-- The quadratic function f(x) = -2(x-3)^2 + 1 -/
def f (x : ℝ) : ℝ := -2 * (x - 3)^2 + 1

/-- The axis of symmetry of f(x) -/
def axis_of_symmetry : ℝ := 3

/-- Theorem: The axis of symmetry of f(x) = -2(x-3)^2 + 1 is x = 3 -/
theorem axis_of_symmetry_is_correct :
  ∀ x : ℝ, f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_is_correct_l1533_153326


namespace NUMINAMATH_CALUDE_soccer_league_games_l1533_153306

/-- The total number of games played in a soccer league. -/
def totalGames (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 12 teams, where each team plays 4 games with each other team,
    the total number of games played is 264. -/
theorem soccer_league_games :
  totalGames 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1533_153306


namespace NUMINAMATH_CALUDE_cl35_properties_neutron_calculation_electron_proton_equality_l1533_153350

/-- Represents an atom with its atomic properties -/
structure Atom where
  protons : ℕ
  mass_number : ℕ
  neutrons : ℕ
  electrons : ℕ

/-- Cl-35 atom -/
def cl35 : Atom :=
  { protons := 17,
    mass_number := 35,
    neutrons := 35 - 17,
    electrons := 17 }

/-- Theorem stating the properties of Cl-35 -/
theorem cl35_properties :
  cl35.protons = 17 ∧
  cl35.mass_number = 35 ∧
  cl35.neutrons = 18 ∧
  cl35.electrons = 17 := by
  sorry

/-- Theorem stating the relationship between neutrons, mass number, and protons -/
theorem neutron_calculation (a : Atom) :
  a.neutrons = a.mass_number - a.protons := by
  sorry

/-- Theorem stating the relationship between electrons and protons -/
theorem electron_proton_equality (a : Atom) :
  a.electrons = a.protons := by
  sorry

end NUMINAMATH_CALUDE_cl35_properties_neutron_calculation_electron_proton_equality_l1533_153350


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l1533_153352

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) : 
  let original_area := 6 * L^2
  let new_edge_length := 1.3 * L
  let new_area := 6 * new_edge_length^2
  (new_area - original_area) / original_area * 100 = 69 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l1533_153352


namespace NUMINAMATH_CALUDE_xyz_product_l1533_153356

theorem xyz_product (x y z : ℕ+) 
  (h1 : x + 2*y = z) 
  (h2 : x^2 - 4*y^2 + z^2 = 310) : 
  x*y*z = 11935 ∨ x*y*z = 2015 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l1533_153356


namespace NUMINAMATH_CALUDE_chickens_per_coop_l1533_153303

/-- Given a farm with chicken coops, prove that the number of chickens per coop is as stated. -/
theorem chickens_per_coop
  (total_coops : ℕ)
  (total_chickens : ℕ)
  (h_coops : total_coops = 9)
  (h_chickens : total_chickens = 540) :
  total_chickens / total_coops = 60 := by
  sorry

end NUMINAMATH_CALUDE_chickens_per_coop_l1533_153303


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l1533_153383

/-- Represents the number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- Represents the number of enchanted stones available. -/
def num_stones : ℕ := 6

/-- Represents the number of herbs that are incompatible with one specific stone. -/
def incompatible_herbs : ℕ := 3

/-- Represents the number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_stones - incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 21 :=
by sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l1533_153383


namespace NUMINAMATH_CALUDE_cultural_group_members_l1533_153338

theorem cultural_group_members :
  ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 200 ∧ n % 7 = 4 ∧ n % 11 = 6 ∧
  (n = 116 ∨ n = 193) :=
by sorry

end NUMINAMATH_CALUDE_cultural_group_members_l1533_153338


namespace NUMINAMATH_CALUDE_smallest_divisible_by_11_ending_in_9_l1533_153331

def is_smallest_divisible_by_11_ending_in_9 (n : ℕ) : Prop :=
  n > 0 ∧ 
  n % 10 = 9 ∧ 
  n % 11 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 10 = 9 → m % 11 = 0 → m ≥ n

theorem smallest_divisible_by_11_ending_in_9 : 
  is_smallest_divisible_by_11_ending_in_9 99 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_11_ending_in_9_l1533_153331


namespace NUMINAMATH_CALUDE_xyz_value_l1533_153313

theorem xyz_value (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (sum_sq_eq : x^2 + y^2 + z^2 = 14)
  (sum_cube_eq : x^3 + y^3 + z^3 = 17) :
  x * y * z = -7 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l1533_153313


namespace NUMINAMATH_CALUDE_new_car_sticker_price_l1533_153369

/-- Calculates the sticker price of a new car based on given conditions --/
theorem new_car_sticker_price 
  (old_car_value : ℝ)
  (old_car_sale_percentage : ℝ)
  (new_car_purchase_percentage : ℝ)
  (out_of_pocket : ℝ)
  (h1 : old_car_value = 20000)
  (h2 : old_car_sale_percentage = 0.8)
  (h3 : new_car_purchase_percentage = 0.9)
  (h4 : out_of_pocket = 11000)
  : ∃ (sticker_price : ℝ), 
    sticker_price * new_car_purchase_percentage - old_car_value * old_car_sale_percentage = out_of_pocket ∧ 
    sticker_price = 30000 := by
  sorry

end NUMINAMATH_CALUDE_new_car_sticker_price_l1533_153369


namespace NUMINAMATH_CALUDE_sanitizer_sprays_effectiveness_l1533_153322

theorem sanitizer_sprays_effectiveness (spray1_kill_rate spray2_kill_rate overlap_rate remaining_rate : Real) :
  spray1_kill_rate = 0.5 →
  overlap_rate = 0.05 →
  remaining_rate = 0.3 →
  1 - (spray1_kill_rate + spray2_kill_rate - overlap_rate) = remaining_rate →
  spray2_kill_rate = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_sanitizer_sprays_effectiveness_l1533_153322


namespace NUMINAMATH_CALUDE_average_weight_b_c_l1533_153318

theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 42 →
  b = 35 →
  (b + c) / 2 = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l1533_153318


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_l1533_153375

/-- Given a triangle with sides 8, 10, and 12, and a similar triangle with perimeter 150,
    prove that the longest side of the similar triangle is 60. -/
theorem similar_triangle_longest_side
  (a b c : ℝ)
  (h_original : a = 8 ∧ b = 10 ∧ c = 12)
  (h_similar_perimeter : ∃ k : ℝ, k * (a + b + c) = 150)
  : ∃ x y z : ℝ, x = k * a ∧ y = k * b ∧ z = k * c ∧ max x (max y z) = 60 :=
sorry

end NUMINAMATH_CALUDE_similar_triangle_longest_side_l1533_153375


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l1533_153343

def is_composite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop :=
  ∀ p, Nat.Prime p → p < 20 → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 529 ∧ has_no_small_prime_factors 529) ∧
  (∀ m : ℕ, m < 529 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l1533_153343


namespace NUMINAMATH_CALUDE_infinite_solutions_l1533_153360

theorem infinite_solutions (a b : ℝ) :
  (∀ x, 4 * (3 * x - a) = 3 * (4 * x + b)) ↔ b = -4/3 * a := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_l1533_153360


namespace NUMINAMATH_CALUDE_inequality_implies_linear_form_l1533_153376

/-- A function satisfying the given inequality for all real x and y -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + y ≤ f (f (f x))

/-- The theorem stating that any function satisfying the inequality must be of the form f(x) = C - x -/
theorem inequality_implies_linear_form {f : ℝ → ℝ} (h : SatisfiesInequality f) :
  ∃ C : ℝ, ∀ x : ℝ, f x = C - x :=
sorry

end NUMINAMATH_CALUDE_inequality_implies_linear_form_l1533_153376


namespace NUMINAMATH_CALUDE_variance_transformation_l1533_153308

variable {n : ℕ}
variable (a : Fin n → ℝ)

def variance (x : Fin n → ℝ) : ℝ := sorry

def transformed_sample (a : Fin n → ℝ) : Fin n → ℝ := 
  fun i => 3 * a i + (if i.val = n - 1 then 2 else 1)

theorem variance_transformation (h : variance a = 3) : 
  variance (transformed_sample a) = 27 := by sorry

end NUMINAMATH_CALUDE_variance_transformation_l1533_153308


namespace NUMINAMATH_CALUDE_ascending_four_digit_difference_l1533_153314

/-- Represents a four-digit number where each subsequent digit is 1 greater than the previous one -/
structure AscendingFourDigitNumber where
  first_digit : ℕ
  constraint : first_digit ≤ 6

/-- Calculates the value of the four-digit number -/
def value (n : AscendingFourDigitNumber) : ℕ :=
  1000 * n.first_digit + 100 * (n.first_digit + 1) + 10 * (n.first_digit + 2) + (n.first_digit + 3)

/-- Calculates the value of the reversed four-digit number -/
def reverse_value (n : AscendingFourDigitNumber) : ℕ :=
  1000 * (n.first_digit + 3) + 100 * (n.first_digit + 2) + 10 * (n.first_digit + 1) + n.first_digit

/-- The main theorem stating that the difference between the reversed number and the original number is always 3087 -/
theorem ascending_four_digit_difference (n : AscendingFourDigitNumber) :
  reverse_value n - value n = 3087 := by
  sorry

end NUMINAMATH_CALUDE_ascending_four_digit_difference_l1533_153314


namespace NUMINAMATH_CALUDE_oplus_problem_l1533_153367

-- Define the operation ⊕
def oplus (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y - a * b

-- State the theorem
theorem oplus_problem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : oplus a b 1 2 = 3) (h2 : oplus a b 2 3 = 6) :
  oplus a b 3 4 = 9 := by sorry

end NUMINAMATH_CALUDE_oplus_problem_l1533_153367


namespace NUMINAMATH_CALUDE_pet_store_dogs_l1533_153325

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
def calculate_dogs (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) : ℕ :=
  (num_cats / cat_ratio) * dog_ratio

/-- Theorem: Given the ratio of cats to dogs is 3:4 and there are 18 cats, there are 24 dogs -/
theorem pet_store_dogs : calculate_dogs 3 4 18 = 24 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l1533_153325


namespace NUMINAMATH_CALUDE_worker_c_completion_time_l1533_153327

/-- The time it takes for worker c to complete a job alone, given the work rates of combinations of workers. -/
theorem worker_c_completion_time 
  (ab_rate : ℚ)  -- Rate at which workers a and b complete the job together
  (abc_rate : ℚ) -- Rate at which workers a, b, and c complete the job together
  (h1 : ab_rate = 1 / 15)  -- a and b finish the job in 15 days
  (h2 : abc_rate = 1 / 5)  -- a, b, and c finish the job in 5 days
  : (1 : ℚ) / (abc_rate - ab_rate) = 15 / 2 := by
  sorry


end NUMINAMATH_CALUDE_worker_c_completion_time_l1533_153327


namespace NUMINAMATH_CALUDE_vector_subtraction_l1533_153354

/-- Given two vectors a and b in ℝ², prove that their difference is equal to a specific vector. -/
theorem vector_subtraction (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (3, 1)) :
  b - a = (2, -1) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1533_153354


namespace NUMINAMATH_CALUDE_a_6_equals_448_l1533_153340

/-- Given a sequence {a_n} where S_n is the sum of its first n terms -/
def S (n : ℕ) : ℕ := n * 2^(n+1)

/-- The nth term of the sequence -/
def a (n : ℕ) : ℕ := S n - S (n-1)

/-- The 6th term of the sequence equals 448 -/
theorem a_6_equals_448 : a 6 = 448 := by sorry

end NUMINAMATH_CALUDE_a_6_equals_448_l1533_153340


namespace NUMINAMATH_CALUDE_unique_room_setup_l1533_153363

/-- Represents the number of people, stools, and chairs in a room -/
structure RoomSetup where
  people : ℕ
  stools : ℕ
  chairs : ℕ

/-- Checks if a given room setup satisfies all conditions -/
def isValidSetup (setup : RoomSetup) : Prop :=
  2 * setup.people + 3 * setup.stools + 4 * setup.chairs = 32 ∧
  setup.people > setup.stools ∧
  setup.people > setup.chairs ∧
  setup.people < setup.stools + setup.chairs

/-- The theorem stating that there is only one valid room setup -/
theorem unique_room_setup :
  ∃! setup : RoomSetup, isValidSetup setup ∧ 
    setup.people = 5 ∧ setup.stools = 2 ∧ setup.chairs = 4 := by
  sorry


end NUMINAMATH_CALUDE_unique_room_setup_l1533_153363


namespace NUMINAMATH_CALUDE_charles_reading_days_l1533_153348

/-- Represents the number of pages Charles reads each day -/
def daily_pages : List Nat := [7, 12, 10, 6]

/-- The total number of pages in the book -/
def total_pages : Nat := 96

/-- Calculates the number of days needed to finish the book -/
def days_to_finish (pages : List Nat) (total : Nat) : Nat :=
  let pages_read := pages.sum
  let remaining := total - pages_read
  let weekdays := pages.length
  let average_daily := (pages_read + remaining - 1) / weekdays
  weekdays + (remaining + average_daily - 1) / average_daily

theorem charles_reading_days :
  days_to_finish daily_pages total_pages = 11 := by
  sorry

#eval days_to_finish daily_pages total_pages

end NUMINAMATH_CALUDE_charles_reading_days_l1533_153348


namespace NUMINAMATH_CALUDE_no_real_solutions_l1533_153341

theorem no_real_solutions : ¬∃ (x : ℝ), 7 * (4 * x + 3) - 4 = -3 * (2 - 9 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1533_153341


namespace NUMINAMATH_CALUDE_line_through_points_l1533_153382

/-- A structure representing a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a line passing through two points -/
def Line (p1 p2 : Point) :=
  {p : Point | (p.y - p1.y) * (p2.x - p1.x) = (p.x - p1.x) * (p2.y - p1.y)}

/-- The statement of the problem -/
theorem line_through_points :
  ∃ (s : Finset ℤ), s.card = 4 ∧
    (∀ m ∈ s, m > 0) ∧
    (∀ m ∈ s, ∃ k : ℤ, k > 0 ∧
      Line (Point.mk (-m) 0) (Point.mk 0 2) (Point.mk 7 k)) ∧
    (∀ m : ℤ, m > 0 →
      (∃ k : ℤ, k > 0 ∧
        Line (Point.mk (-m) 0) (Point.mk 0 2) (Point.mk 7 k)) →
      m ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1533_153382


namespace NUMINAMATH_CALUDE_tom_bought_ten_candies_l1533_153393

/-- Calculates the number of candy pieces Tom bought -/
def candy_bought (initial : ℕ) (from_friend : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial + from_friend)

/-- Theorem stating that Tom bought 10 pieces of candy -/
theorem tom_bought_ten_candies : candy_bought 2 7 19 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tom_bought_ten_candies_l1533_153393


namespace NUMINAMATH_CALUDE_blake_guarantee_four_ruby_prevent_more_than_four_largest_guaranteed_score_l1533_153388

/-- Represents a cell on the infinite grid --/
structure Cell :=
  (x : Int) (y : Int)

/-- Represents the color of a cell --/
inductive Color
  | White
  | Blue
  | Red

/-- Represents the game state --/
structure GameState :=
  (grid : Cell → Color)

/-- Blake's score is the size of the largest blue simple polygon --/
def blakeScore (state : GameState) : Nat :=
  sorry

/-- Blake's strategy to color adjacent cells --/
def blakeStrategy (state : GameState) : Cell :=
  sorry

/-- Ruby's strategy to block Blake --/
def rubyStrategy (state : GameState) : Cell × Cell :=
  sorry

/-- The game play function --/
def playGame (initialState : GameState) : Nat :=
  sorry

theorem blake_guarantee_four :
  ∀ (initialState : GameState),
    (∀ c, initialState.grid c = Color.White) →
    ∃ (finalState : GameState),
      blakeScore finalState ≥ 4 :=
sorry

theorem ruby_prevent_more_than_four :
  ∀ (initialState : GameState),
    (∀ c, initialState.grid c = Color.White) →
    ¬∃ (finalState : GameState),
      blakeScore finalState > 4 :=
sorry

theorem largest_guaranteed_score :
  ∀ (initialState : GameState),
    (∀ c, initialState.grid c = Color.White) →
    (∃ (finalState : GameState), blakeScore finalState = 4) ∧
    (¬∃ (finalState : GameState), blakeScore finalState > 4) :=
sorry

end NUMINAMATH_CALUDE_blake_guarantee_four_ruby_prevent_more_than_four_largest_guaranteed_score_l1533_153388


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l1533_153347

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem largest_perfect_square_factor_of_1800 :
  ∃ (n : ℕ), is_perfect_square n ∧ is_factor n 1800 ∧
  ∀ (m : ℕ), is_perfect_square m → is_factor m 1800 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l1533_153347


namespace NUMINAMATH_CALUDE_average_of_five_numbers_l1533_153396

theorem average_of_five_numbers (numbers : Fin 5 → ℝ) 
  (sum_of_three : ∃ (i j k : Fin 5), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ numbers i + numbers j + numbers k = 48)
  (avg_of_two : ∃ (l m : Fin 5), l ≠ m ∧ (numbers l + numbers m) / 2 = 26) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4) / 5 = 20 := by
sorry

end NUMINAMATH_CALUDE_average_of_five_numbers_l1533_153396


namespace NUMINAMATH_CALUDE_chess_pawns_remaining_l1533_153361

theorem chess_pawns_remaining (initial_pawns : ℕ) 
  (kennedy_lost : ℕ) (riley_lost : ℕ) : 
  initial_pawns = 8 → kennedy_lost = 4 → riley_lost = 1 →
  (initial_pawns - kennedy_lost) + (initial_pawns - riley_lost) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_chess_pawns_remaining_l1533_153361


namespace NUMINAMATH_CALUDE_g_properties_l1533_153372

/-- Given a function f(x) = a - b cos(x) with maximum value 5/2 and minimum value -1/2,
    we define g(x) = -4a sin(bx) and prove its properties. -/
theorem g_properties (a b : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : f = fun x ↦ a - b * Real.cos x)
  (hmax : ∀ x, f x ≤ 5/2)
  (hmin : ∀ x, -1/2 ≤ f x)
  (hg : g = fun x ↦ -4 * a * Real.sin (b * x)) :
  (∃ x, g x = 4) ∧
  (∃ x, g x = -4) ∧
  (∃ T > 0, ∀ x, g (x + T) = g x ∧ ∀ S, 0 < S → S < T → ∃ y, g (y + S) ≠ g y) ∧
  (∀ x, -4 ≤ g x ∧ g x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_g_properties_l1533_153372


namespace NUMINAMATH_CALUDE_sequence_3_9_729_arithmetic_and_geometric_l1533_153357

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- A sequence is geometric if the ratio between consecutive terms is constant -/
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) / a n = r

theorem sequence_3_9_729_arithmetic_and_geometric :
  ∃ (a g : ℕ → ℝ),
    is_arithmetic a ∧ is_geometric g ∧
    (∃ i j k : ℕ, a i = 3 ∧ a j = 9 ∧ a k = 729) ∧
    (∃ x y z : ℕ, g x = 3 ∧ g y = 9 ∧ g z = 729) := by
  sorry

end NUMINAMATH_CALUDE_sequence_3_9_729_arithmetic_and_geometric_l1533_153357


namespace NUMINAMATH_CALUDE_parallel_planes_counterexample_l1533_153395

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (not_parallel : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_counterexample 
  (a b : Line) (α β γ : Plane) : 
  ¬ (∀ (a b : Line) (α β γ : Plane), 
    (subset a α ∧ subset b α ∧ not_parallel a β ∧ not_parallel b β) 
    → ¬(parallel α β)) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_counterexample_l1533_153395


namespace NUMINAMATH_CALUDE_products_sum_bounds_l1533_153362

def CircularArray (α : Type) := Fin 999 → α

def CircularProduct (arr : CircularArray Int) (start : Fin 999) : Int :=
  (List.range 10).foldl (λ acc i => acc * arr ((start + i) % 999)) 1

def SumOfProducts (arr : CircularArray Int) : Int :=
  (List.range 999).foldl (λ acc i => acc + CircularProduct arr i) 0

theorem products_sum_bounds 
  (arr : CircularArray Int) 
  (h1 : ∀ i, arr i = 1 ∨ arr i = -1) 
  (h2 : ∃ i j, arr i ≠ arr j) : 
  -997 ≤ SumOfProducts arr ∧ SumOfProducts arr ≤ 995 :=
sorry

end NUMINAMATH_CALUDE_products_sum_bounds_l1533_153362


namespace NUMINAMATH_CALUDE_gumball_probability_l1533_153387

/-- Given a jar with pink and blue gumballs, where the probability of drawing two blue
    gumballs in a row with replacement is 36/49, the probability of drawing a pink gumball
    is 1/7. -/
theorem gumball_probability (blue pink : ℝ) : 
  blue + pink = 1 →
  blue * blue = 36 / 49 →
  pink = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l1533_153387


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l1533_153397

theorem regular_polygon_interior_angle_sum (n : ℕ) (h1 : n > 2) : 
  (360 / 24 : ℝ) = n → (180 * (n - 2) : ℝ) = 2340 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l1533_153397


namespace NUMINAMATH_CALUDE_equality_of_reciprocals_l1533_153323

theorem equality_of_reciprocals (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (3 : ℝ) ^ a = (4 : ℝ) ^ b ∧ (4 : ℝ) ^ b = (6 : ℝ) ^ c) : 
  2 / c = 2 / a + 1 / b :=
sorry

end NUMINAMATH_CALUDE_equality_of_reciprocals_l1533_153323


namespace NUMINAMATH_CALUDE_tank_capacity_l1533_153346

/-- Represents a cylindrical water tank -/
structure WaterTank where
  capacity : ℝ
  
/-- The tank is 24% full when it contains 72 liters -/
def condition1 (tank : WaterTank) : Prop :=
  0.24 * tank.capacity = 72

/-- The tank is 60% full when it contains 180 liters -/
def condition2 (tank : WaterTank) : Prop :=
  0.60 * tank.capacity = 180

/-- The theorem stating the total capacity of the tank -/
theorem tank_capacity (tank : WaterTank) 
  (h1 : condition1 tank) (h2 : condition2 tank) : 
  tank.capacity = 300 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1533_153346


namespace NUMINAMATH_CALUDE_abc_value_l1533_153381

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 30) (hbc : b * c = 54) (hca : c * a = 45) :
  a * b * c = 270 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l1533_153381


namespace NUMINAMATH_CALUDE_bad_carrots_count_l1533_153330

theorem bad_carrots_count (haley_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ)
  (h1 : haley_carrots = 39)
  (h2 : mom_carrots = 38)
  (h3 : good_carrots = 64) :
  haley_carrots + mom_carrots - good_carrots = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l1533_153330


namespace NUMINAMATH_CALUDE_photo_perimeter_is_23_l1533_153349

/-- Represents a rectangular photograph with a border -/
structure BorderedPhoto where
  width : ℝ
  length : ℝ
  borderWidth : ℝ

/-- Calculates the total area of a bordered photograph -/
def totalArea (photo : BorderedPhoto) : ℝ :=
  (photo.width + 2 * photo.borderWidth) * (photo.length + 2 * photo.borderWidth)

/-- Calculates the perimeter of the photograph without the border -/
def photoPerimeter (photo : BorderedPhoto) : ℝ :=
  2 * (photo.width + photo.length)

theorem photo_perimeter_is_23 (photo : BorderedPhoto) (m : ℝ) :
  photo.borderWidth = 2 →
  totalArea photo = m →
  totalArea { photo with borderWidth := 4 } = m + 94 →
  photoPerimeter photo = 23 := by
  sorry

end NUMINAMATH_CALUDE_photo_perimeter_is_23_l1533_153349


namespace NUMINAMATH_CALUDE_second_replaced_man_age_is_35_l1533_153305

/-- The age of the second replaced man in a group replacement scenario -/
def second_replaced_man_age (initial_count : ℕ) (age_increase : ℕ) 
  (replaced_count : ℕ) (first_replaced_age : ℕ) (new_men_avg_age : ℕ) : ℕ :=
  47 - (initial_count * age_increase)

/-- Theorem stating the age of the second replaced man is 35 -/
theorem second_replaced_man_age_is_35 :
  second_replaced_man_age 12 1 2 21 34 = 35 := by
  sorry

end NUMINAMATH_CALUDE_second_replaced_man_age_is_35_l1533_153305


namespace NUMINAMATH_CALUDE_festival_attendance_l1533_153391

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (h_total : total_students = 1500)
  (h_attendees : festival_attendees = 820) :
  ∃ (girls boys : ℕ),
    girls + boys = total_students ∧
    (3 * girls) / 4 + (2 * boys) / 5 = festival_attendees ∧
    (3 * girls) / 4 = 471 := by
  sorry

end NUMINAMATH_CALUDE_festival_attendance_l1533_153391


namespace NUMINAMATH_CALUDE_x_percent_of_z_l1533_153342

theorem x_percent_of_z (x y z : ℝ) (h1 : x = 1.30 * y) (h2 : y = 0.60 * z) : 
  x = 0.78 * z := by sorry

end NUMINAMATH_CALUDE_x_percent_of_z_l1533_153342


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1533_153370

theorem regular_polygon_exterior_angle (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → (exterior_angle = 30 * Real.pi / 180) → (n * exterior_angle = 2 * Real.pi) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1533_153370


namespace NUMINAMATH_CALUDE_eighth_term_is_21_l1533_153328

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem eighth_term_is_21 :
  fibonacci 7 = 21 ∧ fibonacci 8 = 34 ∧ fibonacci 9 = 55 :=
by sorry

end NUMINAMATH_CALUDE_eighth_term_is_21_l1533_153328


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1533_153345

theorem quadratic_inequality_condition (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) → (-1 ≤ a ∧ a ≤ 3) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1533_153345


namespace NUMINAMATH_CALUDE_robie_chocolates_l1533_153380

theorem robie_chocolates (initial_bags : ℕ) : 
  (initial_bags - 2 + 3 = 4) → initial_bags = 3 := by
  sorry

end NUMINAMATH_CALUDE_robie_chocolates_l1533_153380


namespace NUMINAMATH_CALUDE_edmund_normal_chores_l1533_153392

/-- The number of chores Edmund normally has to do in a week -/
def normal_chores : ℕ := sorry

/-- The number of chores Edmund does per day -/
def chores_per_day : ℕ := 4

/-- The number of days Edmund works -/
def work_days : ℕ := 14

/-- The total amount Edmund earns -/
def total_earnings : ℕ := 64

/-- The payment per extra chore -/
def payment_per_chore : ℕ := 2

theorem edmund_normal_chores :
  normal_chores = 12 :=
by sorry

end NUMINAMATH_CALUDE_edmund_normal_chores_l1533_153392


namespace NUMINAMATH_CALUDE_return_speed_calculation_l1533_153301

/-- Calculates the return speed given the distance, outbound speed, and total time for a round trip -/
theorem return_speed_calculation (distance : ℝ) (outbound_speed : ℝ) (total_time : ℝ) :
  distance = 19.999999999999996 →
  outbound_speed = 25 →
  total_time = 5 + 48 / 60 →
  4 = distance / (total_time - distance / outbound_speed) := by
  sorry

end NUMINAMATH_CALUDE_return_speed_calculation_l1533_153301


namespace NUMINAMATH_CALUDE_probability_centrally_symmetric_shape_l1533_153377

/-- Represents the shapes on the cards -/
inductive Shape
  | Circle
  | Rectangle
  | EquilateralTriangle
  | RegularPentagon

/-- Determines if a shape is centrally symmetric -/
def isCentrallySymmetric (s : Shape) : Bool :=
  match s with
  | Shape.Circle => true
  | Shape.Rectangle => true
  | Shape.EquilateralTriangle => false
  | Shape.RegularPentagon => false

/-- The set of all shapes -/
def allShapes : List Shape :=
  [Shape.Circle, Shape.Rectangle, Shape.EquilateralTriangle, Shape.RegularPentagon]

/-- Theorem: The probability of randomly selecting a centrally symmetric shape is 1/2 -/
theorem probability_centrally_symmetric_shape :
  (allShapes.filter isCentrallySymmetric).length / allShapes.length = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_probability_centrally_symmetric_shape_l1533_153377


namespace NUMINAMATH_CALUDE_sqrt_of_square_positive_l1533_153374

theorem sqrt_of_square_positive (a : ℝ) (h : a > 0) : Real.sqrt (a^2) = a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_square_positive_l1533_153374


namespace NUMINAMATH_CALUDE_sequence_property_l1533_153329

theorem sequence_property (a b c : ℝ) 
  (h1 : (4 * b) ^ 2 = 3 * a * 5 * c)  -- geometric sequence condition
  (h2 : 2 / b = 1 / a + 1 / c)        -- arithmetic sequence condition
  : a / c + c / a = 34 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l1533_153329


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l1533_153335

theorem sugar_solution_percentage (original_percentage : ℝ) : 
  (3/4 : ℝ) * original_percentage + (1/4 : ℝ) * 28 = 16 → 
  original_percentage = 12 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l1533_153335


namespace NUMINAMATH_CALUDE_roots_when_m_zero_m_value_when_product_41_perimeter_of_isosceles_triangle_l1533_153358

-- Define the quadratic equation
def quadratic_eq (m : ℝ) (x : ℝ) : Prop :=
  x^2 - 2*(m+2)*x + m^2 = 0

-- Define the roots of the equation
def roots (m : ℝ) (x₁ x₂ : ℝ) : Prop :=
  quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ x₁ ≠ x₂

-- Theorem for part 1
theorem roots_when_m_zero :
  roots 0 0 4 :=
sorry

-- Theorem for part 2
theorem m_value_when_product_41 :
  ∀ x₁ x₂ : ℝ, roots 9 x₁ x₂ → (x₁ - 2) * (x₂ - 2) = 41 :=
sorry

-- Define an isosceles triangle
def isosceles_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a = b ∨ b = c ∨ a = c)

-- Theorem for part 3
theorem perimeter_of_isosceles_triangle :
  ∀ m x₁ x₂ : ℝ, 
    roots m x₁ x₂ → 
    isosceles_triangle 9 x₁ x₂ → 
    x₁ + x₂ + 9 = 19 :=
sorry

end NUMINAMATH_CALUDE_roots_when_m_zero_m_value_when_product_41_perimeter_of_isosceles_triangle_l1533_153358


namespace NUMINAMATH_CALUDE_circus_performers_standing_time_l1533_153344

/-- The combined time that Pulsar, Polly, and Petra stand on their back legs -/
theorem circus_performers_standing_time :
  let pulsar_time : ℕ := 10
  let polly_time : ℕ := 3 * pulsar_time
  let petra_time : ℕ := polly_time / 6
  pulsar_time + polly_time + petra_time = 45 := by
sorry

end NUMINAMATH_CALUDE_circus_performers_standing_time_l1533_153344


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l1533_153309

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The angle at vertex A of a triangle -/
def angleA (t : Triangle) : ℝ := sorry

/-- Check if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- The region G formed by points P inside the triangle satisfying PA ≤ PB and PA ≤ PC -/
def regionG (t : Triangle) : Set Point :=
  {p : Point | isInside p t ∧ distance p t.A ≤ distance p t.B ∧ distance p t.A ≤ distance p t.C}

/-- The area of region G -/
def areaG (t : Triangle) : ℝ := sorry

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem triangle_is_equilateral (t : Triangle) :
  isAcute t →
  angleA t = π / 3 →
  areaG t = (1 / 3) * triangleArea t →
  isEquilateral t := by sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l1533_153309


namespace NUMINAMATH_CALUDE_expression_result_l1533_153378

theorem expression_result : (3.242 * 12) / 100 = 0.38904 := by
  sorry

end NUMINAMATH_CALUDE_expression_result_l1533_153378


namespace NUMINAMATH_CALUDE_two_numbers_sum_product_l1533_153315

theorem two_numbers_sum_product (S P : ℝ) :
  ∃ (x y : ℝ), x + y = S ∧ x * y = P →
  ((x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
   (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_sum_product_l1533_153315


namespace NUMINAMATH_CALUDE_four_values_with_2001_l1533_153324

/-- Represents a sequence where each term after the first two is defined by the previous two terms. -/
def SpecialSequence (x : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => 2000
  | (n + 2) => SpecialSequence x n * SpecialSequence x (n + 1) - 1

/-- The set of positive real numbers x such that 2001 appears in the special sequence starting with x. -/
def SequencesWith2001 : Set ℝ :=
  {x : ℝ | x > 0 ∧ ∃ n : ℕ, SpecialSequence x n = 2001}

theorem four_values_with_2001 :
  ∃ (S : Finset ℝ), S.card = 4 ∧ (∀ x ∈ SequencesWith2001, x ∈ S) ∧ (∀ x ∈ S, x ∈ SequencesWith2001) :=
sorry

end NUMINAMATH_CALUDE_four_values_with_2001_l1533_153324


namespace NUMINAMATH_CALUDE_fraction_simplification_l1533_153319

theorem fraction_simplification : 
  (3 - 6 + 12 - 24 + 48 - 96) / (6 - 12 + 24 - 48 + 96 - 192) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1533_153319


namespace NUMINAMATH_CALUDE_f_satisfies_data_points_l1533_153353

def f (x : ℝ) : ℝ := 240 - 60 * x

theorem f_satisfies_data_points : 
  (f 0 = 240) ∧ 
  (f 1 = 180) ∧ 
  (f 2 = 120) ∧ 
  (f 3 = 60) ∧ 
  (f 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_data_points_l1533_153353


namespace NUMINAMATH_CALUDE_triangle_cosine_problem_l1533_153384

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b / cos B = c / cos C and cos A = 2/3, then cos B = √6 / 6 -/
theorem triangle_cosine_problem (a b c : ℝ) (A B C : ℝ) :
  b / Real.cos B = c / Real.cos C →
  Real.cos A = 2/3 →
  Real.cos B = Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_problem_l1533_153384


namespace NUMINAMATH_CALUDE_one_weighing_sufficient_l1533_153302

/-- Represents the types of balls -/
inductive BallType
| Aluminum
| Duralumin

/-- The total number of balls -/
def totalBalls : ℕ := 2000

/-- The number of balls in each group -/
def groupSize : ℕ := 1000

/-- The weight of an aluminum ball in grams -/
def aluminumWeight : ℚ := 10

/-- The weight of a duralumin ball in grams -/
def duraluminWeight : ℚ := 9.9

/-- A function that returns the weight of a ball given its type -/
def ballWeight (t : BallType) : ℚ :=
  match t with
  | BallType.Aluminum => aluminumWeight
  | BallType.Duralumin => duraluminWeight

/-- Represents a group of balls -/
structure BallGroup where
  aluminum : ℕ
  duralumin : ℕ

/-- The total weight of a group of balls -/
def groupWeight (g : BallGroup) : ℚ :=
  g.aluminum * aluminumWeight + g.duralumin * duraluminWeight

/-- Theorem stating that it's possible to separate the balls into two groups
    with equal size but different weights using one weighing -/
theorem one_weighing_sufficient :
  ∃ (g1 g2 : BallGroup),
    g1.aluminum + g1.duralumin = groupSize ∧
    g2.aluminum + g2.duralumin = groupSize ∧
    g1.aluminum + g2.aluminum = groupSize ∧
    g1.duralumin + g2.duralumin = groupSize ∧
    groupWeight g1 ≠ groupWeight g2 :=
  sorry

end NUMINAMATH_CALUDE_one_weighing_sufficient_l1533_153302


namespace NUMINAMATH_CALUDE_cards_left_l1533_153390

/-- The number of basketball card boxes Ben has -/
def basketball_boxes : ℕ := 4

/-- The number of cards in each basketball box -/
def basketball_cards_per_box : ℕ := 10

/-- The number of baseball card boxes Ben's mother gave him -/
def baseball_boxes : ℕ := 5

/-- The number of cards in each baseball box -/
def baseball_cards_per_box : ℕ := 8

/-- The number of cards Ben gave to his classmates -/
def cards_given_away : ℕ := 58

/-- Theorem stating the number of cards Ben has left -/
theorem cards_left : 
  basketball_boxes * basketball_cards_per_box + 
  baseball_boxes * baseball_cards_per_box - 
  cards_given_away = 22 := by sorry

end NUMINAMATH_CALUDE_cards_left_l1533_153390


namespace NUMINAMATH_CALUDE_club_group_size_l1533_153398

theorem club_group_size (N : ℕ) (x : ℕ) 
  (h1 : 10 < N ∧ N < 40)
  (h2 : (N - 3) % 5 = 0 ∧ (N - 3) % 6 = 0)
  (h3 : N % x = 5)
  : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_club_group_size_l1533_153398


namespace NUMINAMATH_CALUDE_two_complex_roots_iff_m_values_l1533_153310

/-- The equation (x / (x+2)) + (x / (x+3)) = mx has exactly two complex roots
    if and only if m is equal to 0, 2i, or -2i. -/
theorem two_complex_roots_iff_m_values (m : ℂ) : 
  (∃! (r₁ r₂ : ℂ), ∀ (x : ℂ), x ≠ -2 ∧ x ≠ -3 →
    (x / (x + 2) + x / (x + 3) = m * x) ↔ (x = r₁ ∨ x = r₂)) ↔
  (m = 0 ∨ m = 2*I ∨ m = -2*I) :=
sorry

end NUMINAMATH_CALUDE_two_complex_roots_iff_m_values_l1533_153310


namespace NUMINAMATH_CALUDE_share_division_l1533_153339

/-- Given a total sum to be divided among three people A, B, and C, where
    3 times A's share equals 4 times B's share equals 7 times C's share,
    prove that C's share is 84 when the total sum is 427. -/
theorem share_division (total : ℕ) (a b c : ℚ)
  (h_total : total = 427)
  (h_sum : a + b + c = total)
  (h_prop : 3 * a = 4 * b ∧ 4 * b = 7 * c) :
  c = 84 := by
  sorry

end NUMINAMATH_CALUDE_share_division_l1533_153339


namespace NUMINAMATH_CALUDE_relationship_abc_l1533_153332

theorem relationship_abc : 
  let a := Real.sin (15 * π / 180) * Real.cos (15 * π / 180)
  let b := Real.cos (π / 6) ^ 2 - Real.sin (π / 6) ^ 2
  let c := Real.tan (30 * π / 180) / (1 - Real.tan (30 * π / 180) ^ 2)
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1533_153332


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l1533_153394

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel 
  (b c : Line) (α β : Plane) :
  perpendicular c β → parallel c α → plane_perpendicular α β :=
by sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l1533_153394


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_2018_l1533_153316

theorem sum_of_x_and_y_is_2018 (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x^4 - 2018*x^3 - 2018*y^2*x = y^4 - 2018*y^3 - 2018*y*x^2) : 
  x + y = 2018 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_2018_l1533_153316
