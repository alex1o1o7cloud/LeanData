import Mathlib

namespace NUMINAMATH_GPT_total_full_parking_spots_correct_l1528_152864

-- Define the number of parking spots on each level
def total_parking_spots (level : ℕ) : ℕ :=
  100 + (level - 1) * 50

-- Define the number of open spots on each level
def open_parking_spots (level : ℕ) : ℕ :=
  if level = 1 then 58
  else if level <= 4 then 58 - 3 * (level - 1)
  else 49 + 10 * (level - 4)

-- Define the number of full parking spots on each level
def full_parking_spots (level : ℕ) : ℕ :=
  total_parking_spots level - open_parking_spots level

-- Sum up the full parking spots on all 7 levels to get the total full spots
def total_full_parking_spots : ℕ :=
  List.sum (List.map full_parking_spots [1, 2, 3, 4, 5, 6, 7])

-- Theorem to prove the total number of full parking spots
theorem total_full_parking_spots_correct : total_full_parking_spots = 1329 :=
by
  sorry

end NUMINAMATH_GPT_total_full_parking_spots_correct_l1528_152864


namespace NUMINAMATH_GPT_neg_p_false_sufficient_but_not_necessary_for_p_or_q_l1528_152861

variable (p q : Prop)

theorem neg_p_false_sufficient_but_not_necessary_for_p_or_q :
  (¬ p = false) → (p ∨ q) ∧ ¬((p ∨ q) → (¬ p = false)) :=
by
  sorry

end NUMINAMATH_GPT_neg_p_false_sufficient_but_not_necessary_for_p_or_q_l1528_152861


namespace NUMINAMATH_GPT_pyramids_from_cuboid_l1528_152872

-- Define the vertices of a cuboid
def vertices_of_cuboid : ℕ := 8

-- Define the edges of a cuboid
def edges_of_cuboid : ℕ := 12

-- Define the faces of a cuboid
def faces_of_cuboid : ℕ := 6

-- Define the combinatoric calculation
def combinations (n k : ℕ) : ℕ := (n.choose k)

-- Define the total number of tetrahedrons formed
def total_tetrahedrons : ℕ := combinations 7 3 - faces_of_cuboid * combinations 4 3

-- Define the expected result
def expected_tetrahedrons : ℕ := 106

-- The theorem statement to prove that the total number of tetrahedrons is 106
theorem pyramids_from_cuboid : total_tetrahedrons = expected_tetrahedrons :=
by
  sorry

end NUMINAMATH_GPT_pyramids_from_cuboid_l1528_152872


namespace NUMINAMATH_GPT_prism_lateral_edges_correct_cone_axial_section_equilateral_l1528_152816

/-- Defining the lateral edges of a prism and its properties --/
structure Prism (r : ℝ) :=
(lateral_edges_equal : ∀ (e1 e2 : ℝ), e1 = r ∧ e2 = r)

/-- Defining the axial section of a cone with properties of base radius and generatrix length --/
structure Cone (r : ℝ) :=
(base_radius : ℝ := r)
(generatrix_length : ℝ := 2 * r)
(is_equilateral : base_radius * 2 = generatrix_length)

theorem prism_lateral_edges_correct (r : ℝ) (P : Prism r) : 
 ∃ e, e = r ∧ ∀ e', e' = r :=
by {
  sorry
}

theorem cone_axial_section_equilateral (r : ℝ) (C : Cone r) : 
 base_radius * 2 = generatrix_length :=
by {
  sorry
}

end NUMINAMATH_GPT_prism_lateral_edges_correct_cone_axial_section_equilateral_l1528_152816


namespace NUMINAMATH_GPT_differentiable_additive_zero_derivative_l1528_152880

theorem differentiable_additive_zero_derivative {f : ℝ → ℝ}
  (h1 : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
  (h_diff : Differentiable ℝ f) : 
  deriv f 0 = 0 :=
sorry

end NUMINAMATH_GPT_differentiable_additive_zero_derivative_l1528_152880


namespace NUMINAMATH_GPT_max_children_typeA_max_children_typeB_max_children_typeC_max_children_typeD_l1528_152890

structure BusConfig where
  rows_section1 : ℕ
  seats_per_row_section1 : ℕ
  rows_section2 : ℕ
  seats_per_row_section2 : ℕ
  total_seats : ℕ
  max_children : ℕ

def typeA : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 4,
    rows_section2 := 4,
    seats_per_row_section2 := 3,
    total_seats := 36,
    max_children := 40 }

def typeB : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 4,
    rows_section2 := 6,
    seats_per_row_section2 := 5,
    total_seats := 54,
    max_children := 50 }

def typeC : BusConfig :=
  { rows_section1 := 8,
    seats_per_row_section1 := 4,
    rows_section2 := 2,
    seats_per_row_section2 := 2,
    total_seats := 36,
    max_children := 35 }

def typeD : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 3,
    rows_section2 := 6,
    seats_per_row_section2 := 3,
    total_seats := 36,
    max_children := 30 }

theorem max_children_typeA : min typeA.total_seats typeA.max_children = 36 := by
  sorry

theorem max_children_typeB : min typeB.total_seats typeB.max_children = 50 := by
  sorry

theorem max_children_typeC : min typeC.total_seats typeC.max_children = 35 := by
  sorry

theorem max_children_typeD : min typeD.total_seats typeD.max_children = 30 := by
  sorry

end NUMINAMATH_GPT_max_children_typeA_max_children_typeB_max_children_typeC_max_children_typeD_l1528_152890


namespace NUMINAMATH_GPT_minimum_road_length_l1528_152899

/-- Define the grid points A, B, and C with their coordinates. -/
def A : ℤ × ℤ := (0, 0)
def B : ℤ × ℤ := (3, 2)
def C : ℤ × ℤ := (4, 3)

/-- Define the side length of each grid square in meters. -/
def side_length : ℕ := 100

/-- Calculate the Manhattan distance between two points on the grid. -/
def manhattan_distance (p q : ℤ × ℤ) : ℕ :=
  (Int.natAbs (p.1 - q.1) + Int.natAbs (p.2 - q.2)) * side_length

/-- Statement: The minimum total length of the roads (in meters) to connect A, B, and C is 1000 meters. -/
theorem minimum_road_length : manhattan_distance A B + manhattan_distance B C + manhattan_distance C A = 1000 := by
  sorry

end NUMINAMATH_GPT_minimum_road_length_l1528_152899


namespace NUMINAMATH_GPT_b_present_age_l1528_152898

/-- 
In 10 years, A will be twice as old as B was 10 years ago. 
A is currently 8 years older than B. 
Prove that B's current age is 38.
--/
theorem b_present_age (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10)) 
  (h2 : a = b + 8) : 
  b = 38 := 
  sorry

end NUMINAMATH_GPT_b_present_age_l1528_152898


namespace NUMINAMATH_GPT_perpendicular_bisector_l1528_152865

theorem perpendicular_bisector (x y : ℝ) :
  (x - 2 * y + 1 = 0 ∧ -1 ≤ x ∧ x ≤ 3) → (2 * x + y - 3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_bisector_l1528_152865


namespace NUMINAMATH_GPT_luke_can_see_silvia_for_22_point_5_minutes_l1528_152838

/--
Luke is initially 0.75 miles behind Silvia. Luke rollerblades at 10 mph and Silvia cycles 
at 6 mph. Luke can see Silvia until she is 0.75 miles behind him. Prove that Luke can see 
Silvia for a total of 22.5 minutes.
-/
theorem luke_can_see_silvia_for_22_point_5_minutes :
    let distance := (3 / 4 : ℝ)
    let luke_speed := (10 : ℝ)
    let silvia_speed := (6 : ℝ)
    let relative_speed := luke_speed - silvia_speed
    let time_to_reach := distance / relative_speed
    let total_time := 2 * time_to_reach * 60 
    total_time = 22.5 :=
by
    sorry

end NUMINAMATH_GPT_luke_can_see_silvia_for_22_point_5_minutes_l1528_152838


namespace NUMINAMATH_GPT_find_intersection_l1528_152887

open Set Real

def domain_A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def domain_B : Set ℝ := {x : ℝ | x < 1}

def intersection (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem find_intersection :
  intersection domain_A domain_B = {x : ℝ | -2 ≤ x ∧ x < 1} := 
by sorry

end NUMINAMATH_GPT_find_intersection_l1528_152887


namespace NUMINAMATH_GPT_minimum_games_pasha_wins_l1528_152809

noncomputable def pasha_initial_money : Nat := 9 -- Pasha has a single-digit amount
noncomputable def igor_initial_money : Nat := 1000 -- Igor has a four-digit amount
noncomputable def pasha_final_money : Nat := 100 -- Pasha has a three-digit amount
noncomputable def igor_final_money : Nat := 99 -- Igor has a two-digit amount

theorem minimum_games_pasha_wins :
  ∃ (games_won_by_pasha : Nat), 
    (games_won_by_pasha >= 7) ∧
    (games_won_by_pasha <= 7) := sorry

end NUMINAMATH_GPT_minimum_games_pasha_wins_l1528_152809


namespace NUMINAMATH_GPT_height_of_parallelogram_l1528_152866

theorem height_of_parallelogram (A B h : ℝ) (hA : A = 72) (hB : B = 12) (h_area : A = B * h) : h = 6 := by
  sorry

end NUMINAMATH_GPT_height_of_parallelogram_l1528_152866


namespace NUMINAMATH_GPT_relative_speed_of_trains_l1528_152835

def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600

theorem relative_speed_of_trains 
  (speed_train1_kmph : ℕ) 
  (speed_train2_kmph : ℕ) 
  (h1 : speed_train1_kmph = 216) 
  (h2 : speed_train2_kmph = 180) : 
  kmph_to_mps speed_train1_kmph - kmph_to_mps speed_train2_kmph = 10 := 
by 
  sorry

end NUMINAMATH_GPT_relative_speed_of_trains_l1528_152835


namespace NUMINAMATH_GPT_triangle_cot_tan_identity_l1528_152882

theorem triangle_cot_tan_identity 
  (a b c : ℝ) 
  (h : a^2 + b^2 = 2018 * c^2)
  (A B C : ℝ) 
  (triangle_ABC : ∀ (a b c : ℝ), a + b + c = π) 
  (cot_A : ℝ := Real.cos A / Real.sin A) 
  (cot_B : ℝ := Real.cos B / Real.sin B) 
  (tan_C : ℝ := Real.sin C / Real.cos C) :
  (cot_A + cot_B) * tan_C = -2 / 2017 :=
by sorry

end NUMINAMATH_GPT_triangle_cot_tan_identity_l1528_152882


namespace NUMINAMATH_GPT_relationship_among_values_l1528_152896

-- Define the properties of the function f
variables (f : ℝ → ℝ)

-- Assume necessary conditions
axiom domain_of_f : ∀ x : ℝ, f x ≠ 0 -- Domain of f is ℝ
axiom even_function : ∀ x : ℝ, f (-x) = f x -- f is an even function
axiom increasing_function : ∀ x y : ℝ, (0 ≤ x) → (x ≤ y) → (f x ≤ f y) -- f is increasing for x in [0, + ∞)

-- Define the main theorem based on the problem statement
theorem relationship_among_values : f π > f (-3) ∧ f (-3) > f (-2) :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_values_l1528_152896


namespace NUMINAMATH_GPT_base9_minus_base6_l1528_152819

-- Definitions from conditions
def base9_to_base10 (n : Nat) : Nat :=
  match n with
  | 325 => 3 * 9^2 + 2 * 9^1 + 5 * 9^0
  | _ => 0

def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 231 => 2 * 6^2 + 3 * 6^1 + 1 * 6^0
  | _ => 0

-- Main theorem statement
theorem base9_minus_base6 : base9_to_base10 325 - base6_to_base10 231 = 175 :=
by
  sorry

end NUMINAMATH_GPT_base9_minus_base6_l1528_152819


namespace NUMINAMATH_GPT_number_of_girls_in_class_l1528_152881

section
variables (g b : ℕ)

/-- Given the total number of students and the ratio of girls to boys, this theorem states the number of girls in Ben's class. -/
theorem number_of_girls_in_class (h1 : 3 * b = 4 * g) (h2 : g + b = 35) : g = 15 :=
sorry
end

end NUMINAMATH_GPT_number_of_girls_in_class_l1528_152881


namespace NUMINAMATH_GPT_solve_real_number_pairs_l1528_152867

theorem solve_real_number_pairs (x y : ℝ) :
  (x^2 + y^2 - 48 * x - 29 * y + 714 = 0 ∧ 2 * x * y - 29 * x - 48 * y + 756 = 0) ↔
  (x = 31.5 ∧ y = 10.5) ∨ (x = 20 ∧ y = 22) ∨ (x = 28 ∧ y = 7) ∨ (x = 16.5 ∧ y = 18.5) :=
by
  sorry

end NUMINAMATH_GPT_solve_real_number_pairs_l1528_152867


namespace NUMINAMATH_GPT_smallest_tree_height_correct_l1528_152833

-- Defining the conditions
def TallestTreeHeight : ℕ := 108
def MiddleTreeHeight (tallest : ℕ) : ℕ := (tallest / 2) - 6
def SmallestTreeHeight (middle : ℕ) : ℕ := middle / 4

-- Proof statement
theorem smallest_tree_height_correct :
  SmallestTreeHeight (MiddleTreeHeight TallestTreeHeight) = 12 :=
by
  -- Here we would put the proof, but we are skipping it with sorry.
  sorry

end NUMINAMATH_GPT_smallest_tree_height_correct_l1528_152833


namespace NUMINAMATH_GPT_ab_power_2023_l1528_152839

theorem ab_power_2023 (a b : ℤ) (h : |a + 2| + (b - 1) ^ 2 = 0) : (a + b) ^ 2023 = -1 :=
by
  sorry

end NUMINAMATH_GPT_ab_power_2023_l1528_152839


namespace NUMINAMATH_GPT_worker_hourly_rate_l1528_152895

theorem worker_hourly_rate (x : ℝ) (h1 : 8 * 0.90 = 7.20) (h2 : 42 * x + 7.20 = 32.40) : x = 0.60 :=
by
  sorry

end NUMINAMATH_GPT_worker_hourly_rate_l1528_152895


namespace NUMINAMATH_GPT_women_in_club_l1528_152817

theorem women_in_club (total_members : ℕ) (men : ℕ) (total_members_eq : total_members = 52) (men_eq : men = 37) :
  ∃ women : ℕ, women = 15 :=
by
  sorry

end NUMINAMATH_GPT_women_in_club_l1528_152817


namespace NUMINAMATH_GPT_Ada_initial_seat_l1528_152855

-- We have 6 seats
def Seats := Fin 6

-- Friends' movements expressed in terms of seat positions changes
variable (Bea Ceci Dee Edie Fred Ada : Seats)

-- Conditions about the movements
variable (beMovedRight : Bea.val + 1 = Ada.val)
variable (ceMovedLeft : Ceci.val = Ada.val + 2)
variable (deeMovedRight : Dee.val + 1 = Ada.val)
variable (edieFredSwitch : ∀ (edie_new fred_new : Seats), 
  edie_new = Fred ∧ fred_new = Edie)

-- Ada returns to an end seat (1 or 6)
axiom adaEndSeat : Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩

-- Theorem to prove Ada's initial position
theorem Ada_initial_seat (Bea Ceci Dee Edie Fred Ada : Seats)
  (beMovedRight : Bea.val + 1 = Ada.val)
  (ceMovedLeft : Ceci.val = Ada.val + 2)
  (deeMovedRight : Dee.val + 1 = Ada.val)
  (edieFredSwitch : ∀ (edie_new fred_new : Seats), 
    edie_new = Fred ∧ fred_new = Edie)
  (adaEndSeat : Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩) :
  Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩ := sorry

end NUMINAMATH_GPT_Ada_initial_seat_l1528_152855


namespace NUMINAMATH_GPT_total_stocks_l1528_152849

-- Define the conditions as given in the math problem
def closed_higher : ℕ := 1080
def ratio : ℝ := 1.20

-- Using ℕ for the number of stocks that closed lower
def closed_lower (x : ℕ) : Prop := 1080 = x * ratio ∧ closed_higher = x + x * (1 / 5)

-- Definition to compute the total number of stocks on the stock exchange
def total_number_of_stocks (x : ℕ) : ℕ := closed_higher + x

-- The main theorem to be proved
theorem total_stocks (x : ℕ) (h : closed_lower x) : total_number_of_stocks x = 1980 :=
sorry

end NUMINAMATH_GPT_total_stocks_l1528_152849


namespace NUMINAMATH_GPT_sum_of_digits_of_power_eight_2010_l1528_152850

theorem sum_of_digits_of_power_eight_2010 :
  let n := 2010
  let a := 8
  let tens_digit := (a ^ n / 10) % 10
  let units_digit := a ^ n % 10
  tens_digit + units_digit = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_power_eight_2010_l1528_152850


namespace NUMINAMATH_GPT_trigonometric_identity_l1528_152875

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10)) / (Real.sin (α - Real.pi / 5)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1528_152875


namespace NUMINAMATH_GPT_bus_fare_with_train_change_in_total_passengers_l1528_152831

variables (p : ℝ) (q : ℝ) (TC : ℝ → ℝ)
variables (p_train : ℝ) (train_capacity : ℝ)

-- Demand function
def demand_function (p : ℝ) : ℝ := 4200 - 100 * p

-- Train fare is fixed
def train_fare : ℝ := 4

-- Train capacity
def train_cap : ℝ := 800

-- Bus total cost function
def total_cost (y : ℝ) : ℝ := 10 * y + 225

-- Case when there is competition (train available)
def optimal_bus_fare_with_train : ℝ := 22

-- Case when there is no competition (train service is closed)
def optimal_bus_fare_without_train : ℝ := 26

-- Change in the number of passengers when the train service closes
def change_in_passengers : ℝ := 400

-- Theorems to prove
theorem bus_fare_with_train : optimal_bus_fare_with_train = 22 := sorry
theorem change_in_total_passengers : change_in_passengers = 400 := sorry

end NUMINAMATH_GPT_bus_fare_with_train_change_in_total_passengers_l1528_152831


namespace NUMINAMATH_GPT_max_min_values_l1528_152832

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values :
  (∀ x ∈ (Set.Icc 0 2), f x ≤ 5) ∧ (∃ x ∈ (Set.Icc 0 2), f x = 5) ∧
  (∀ x ∈ (Set.Icc 0 2), f x ≥ -15) ∧ (∃ x ∈ (Set.Icc 0 2), f x = -15) :=
by
  sorry

end NUMINAMATH_GPT_max_min_values_l1528_152832


namespace NUMINAMATH_GPT_incorrect_statement_D_l1528_152856

theorem incorrect_statement_D (k b x : ℝ) (hk : k < 0) (hb : b > 0) (hx : x > -b / k) :
  k * x + b ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_D_l1528_152856


namespace NUMINAMATH_GPT_psychologist_charge_difference_l1528_152825

-- Define the variables and conditions
variables (F A : ℝ)
axiom cond1 : F + 4 * A = 250
axiom cond2 : F + A = 115

theorem psychologist_charge_difference : F - A = 25 :=
by
  -- conditions are already stated as axioms, we'll just provide the target theorem
  sorry

end NUMINAMATH_GPT_psychologist_charge_difference_l1528_152825


namespace NUMINAMATH_GPT_matchstick_triangles_l1528_152807

theorem matchstick_triangles (perimeter : ℕ) (h_perimeter : perimeter = 30) : 
  ∃ n : ℕ, n = 17 ∧ 
  (∀ a b c : ℕ, a + b + c = perimeter → a > 0 → b > 0 → c > 0 → 
                a + b > c ∧ a + c > b ∧ b + c > a → 
                a ≤ b ∧ b ≤ c → n = 17) := 
sorry

end NUMINAMATH_GPT_matchstick_triangles_l1528_152807


namespace NUMINAMATH_GPT_compute_expression_l1528_152888

theorem compute_expression : 2 + 5 * 3 - 4 + 6 * 2 / 3 = 17 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1528_152888


namespace NUMINAMATH_GPT_range_of_a_l1528_152818

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3/5 < a ∧ a ≤ 1) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1528_152818


namespace NUMINAMATH_GPT_kim_hours_of_classes_per_day_l1528_152893

-- Definitions based on conditions
def original_classes : Nat := 4
def hours_per_class : Nat := 2
def dropped_classes : Nat := 1

-- Prove that Kim now has 6 hours of classes per day
theorem kim_hours_of_classes_per_day : (original_classes - dropped_classes) * hours_per_class = 6 := by
  sorry

end NUMINAMATH_GPT_kim_hours_of_classes_per_day_l1528_152893


namespace NUMINAMATH_GPT_number_from_division_l1528_152826

theorem number_from_division (number : ℝ) (h : number / 2000 = 0.012625) : number = 25.25 :=
by
  sorry

end NUMINAMATH_GPT_number_from_division_l1528_152826


namespace NUMINAMATH_GPT_no_real_solution_ineq_l1528_152804

theorem no_real_solution_ineq (x : ℝ) (h : x ≠ 5) : ¬ (x^3 - 125) / (x - 5) < 0 := 
by
  sorry

end NUMINAMATH_GPT_no_real_solution_ineq_l1528_152804


namespace NUMINAMATH_GPT_valentine_floral_requirement_l1528_152891

theorem valentine_floral_requirement:
  let nursing_home_roses := 90
  let nursing_home_tulips := 80
  let nursing_home_lilies := 100
  let shelter_roses := 120
  let shelter_tulips := 75
  let shelter_lilies := 95
  let maternity_ward_roses := 100
  let maternity_ward_tulips := 110
  let maternity_ward_lilies := 85
  let total_roses := nursing_home_roses + shelter_roses + maternity_ward_roses
  let total_tulips := nursing_home_tulips + shelter_tulips + maternity_ward_tulips
  let total_lilies := nursing_home_lilies + shelter_lilies + maternity_ward_lilies
  let total_flowers := total_roses + total_tulips + total_lilies
  total_roses = 310 ∧
  total_tulips = 265 ∧
  total_lilies = 280 ∧
  total_flowers = 855 :=
by
  sorry

end NUMINAMATH_GPT_valentine_floral_requirement_l1528_152891


namespace NUMINAMATH_GPT_outfit_combination_count_l1528_152814

theorem outfit_combination_count (c : ℕ) (s p h sh : ℕ) (c_eq_6 : c = 6) (s_eq_c : s = c) (p_eq_c : p = c) (h_eq_c : h = c) (sh_eq_c : sh = c) :
  (c^4) - c = 1290 :=
by
  sorry

end NUMINAMATH_GPT_outfit_combination_count_l1528_152814


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_l1528_152892

open Nat

noncomputable def arithmetic_sum (a1 d n : ℕ) : ℝ :=
  (2 * a1 + (n - 1) * d) * n / 2

theorem sum_arithmetic_sequence (m n : ℕ) (h1 : m ≠ n) (h2 : m > 0) (h3 : n > 0)
    (S_m S_n : ℝ) (h4 : S_m = m / n) (h5 : S_n = n / m) 
    (a1 d : ℕ) (h6 : S_m = arithmetic_sum a1 d m) (h7 : S_n = arithmetic_sum a1 d n) 
    : arithmetic_sum a1 d (m + n) > 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_l1528_152892


namespace NUMINAMATH_GPT_metallic_sheet_first_dimension_l1528_152822

-- Given Conditions
variable (x : ℝ) (height width : ℝ)
def metallic_sheet :=
  (x > 0) ∧ (height = 8) ∧ (width = 36 - 2 * height)

-- Volume of the resulting box should be 5760 m³
def volume_box :=
  (width - 2 * height) * (x - 2 * height) * height = 5760

-- Prove the first dimension of the metallic sheet
theorem metallic_sheet_first_dimension (h1 : metallic_sheet x height width) (h2 : volume_box x height width) : 
  x = 52 :=
  sorry

end NUMINAMATH_GPT_metallic_sheet_first_dimension_l1528_152822


namespace NUMINAMATH_GPT_rate_per_meter_for_fencing_l1528_152854

/-- The length of a rectangular plot is 10 meters more than its width. 
    The cost of fencing the plot along its perimeter at a certain rate per meter is Rs. 1430. 
    The perimeter of the plot is 220 meters. 
    Prove that the rate per meter for fencing the plot is 6.5 Rs. 
 -/
theorem rate_per_meter_for_fencing (width length perimeter cost : ℝ)
  (h_length : length = width + 10)
  (h_perimeter : perimeter = 2 * (width + length))
  (h_perimeter_value : perimeter = 220)
  (h_cost : cost = 1430) :
  (cost / perimeter) = 6.5 := by
  sorry

end NUMINAMATH_GPT_rate_per_meter_for_fencing_l1528_152854


namespace NUMINAMATH_GPT_inequality_for_positive_reals_l1528_152869

variable {a b c : ℝ}
variable {k : ℕ}

theorem inequality_for_positive_reals 
  (hab : a > 0) 
  (hbc : b > 0) 
  (hac : c > 0) 
  (hprod : a * b * c = 1) 
  (hk : k ≥ 2) 
  : (a ^ k) / (a + b) + (b ^ k) / (b + c) + (c ^ k) / (c + a) ≥ 3 / 2 := 
sorry

end NUMINAMATH_GPT_inequality_for_positive_reals_l1528_152869


namespace NUMINAMATH_GPT_expression_evaluation_l1528_152860

variable (a b : ℝ)

theorem expression_evaluation (h : a + b = 1) :
  a^3 + b^3 + 3 * (a^3 * b + a * b^3) + 6 * (a^3 * b^2 + a^2 * b^3) = 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1528_152860


namespace NUMINAMATH_GPT_fourth_graders_bought_more_markers_l1528_152837

-- Define the conditions
def cost_per_marker : ℕ := 20
def total_payment_fifth_graders : ℕ := 180
def total_payment_fourth_graders : ℕ := 200

-- Compute the number of markers bought by fifth and fourth graders
def markers_bought_by_fifth_graders : ℕ := total_payment_fifth_graders / cost_per_marker
def markers_bought_by_fourth_graders : ℕ := total_payment_fourth_graders / cost_per_marker

-- Statement to prove
theorem fourth_graders_bought_more_markers : 
  markers_bought_by_fourth_graders - markers_bought_by_fifth_graders = 1 := by
  sorry

end NUMINAMATH_GPT_fourth_graders_bought_more_markers_l1528_152837


namespace NUMINAMATH_GPT_sum_of_remainders_l1528_152878

theorem sum_of_remainders (a b c d : ℕ) 
  (h1 : a % 13 = 3) 
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7)
  (h4 : d % 13 = 9) : 
  (a + b + c + d) % 13 = 11 := 
by {
  sorry -- Proof not required as per instructions
}

end NUMINAMATH_GPT_sum_of_remainders_l1528_152878


namespace NUMINAMATH_GPT_calculate_outlet_requirements_l1528_152877

def outlets_needed := 10
def suites_outlets_needed := 15
def num_standard_rooms := 50
def num_suites := 10
def type_a_percentage := 0.40
def type_b_percentage := 0.60
def type_c_percentage := 1.0

noncomputable def total_outlets_needed := 500 + 150
noncomputable def type_a_outlets_needed := 0.40 * 500
noncomputable def type_b_outlets_needed := 0.60 * 500
noncomputable def type_c_outlets_needed := 150

theorem calculate_outlet_requirements :
  total_outlets_needed = 650 ∧
  type_a_outlets_needed = 200 ∧
  type_b_outlets_needed = 300 ∧
  type_c_outlets_needed = 150 :=
by
  sorry

end NUMINAMATH_GPT_calculate_outlet_requirements_l1528_152877


namespace NUMINAMATH_GPT_find_second_game_points_l1528_152876

-- Define Clayton's points for respective games
def first_game_points := 10
def third_game_points := 6

-- Define the points in the second game as P
variable (P : ℕ)

-- Define the points in the fourth game based on the average of first three games
def fourth_game_points := (first_game_points + P + third_game_points) / 3

-- Define the total points over four games
def total_points := first_game_points + P + third_game_points + fourth_game_points

-- Based on the total points, prove P = 14
theorem find_second_game_points (P : ℕ) (h : total_points P = 40) : P = 14 :=
  by
    sorry

end NUMINAMATH_GPT_find_second_game_points_l1528_152876


namespace NUMINAMATH_GPT_new_average_of_adjusted_consecutive_integers_l1528_152897

theorem new_average_of_adjusted_consecutive_integers
  (x : ℝ)
  (h1 : (1 / 10) * (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) = 25)
  : (1 / 10) * ((x - 9) + (x + 1 - 8) + (x + 2 - 7) + (x + 3 - 6) + (x + 4 - 5) + (x + 5 - 4) + (x + 6 - 3) + (x + 7 - 2) + (x + 8 - 1) + (x + 9 - 0)) = 20.5 := 
by sorry

end NUMINAMATH_GPT_new_average_of_adjusted_consecutive_integers_l1528_152897


namespace NUMINAMATH_GPT_A_eq_B_l1528_152879

open Set

def A := {x | ∃ a : ℝ, x = 5 - 4 * a + a ^ 2}
def B := {y | ∃ b : ℝ, y = 4 * b ^ 2 + 4 * b + 2}

theorem A_eq_B : A = B := sorry

end NUMINAMATH_GPT_A_eq_B_l1528_152879


namespace NUMINAMATH_GPT_book_pricing_and_min_cost_l1528_152841

-- Define the conditions
def price_relation (a : ℝ) (ps_price : ℝ) : Prop :=
  ps_price = 1.2 * a

def book_count_relation (a : ℝ) (lit_count ps_count : ℕ) : Prop :=
  lit_count = 1200 / a ∧ ps_count = 1200 / (1.2 * a) ∧ lit_count - ps_count = 10

def min_cost_condition (x : ℕ) : Prop :=
  x ≤ 600

def total_cost (x : ℕ) : ℝ :=
  20 * x + 24 * (1000 - x)

-- The theorem combining all parts
theorem book_pricing_and_min_cost:
  ∃ (a : ℝ) (ps_price : ℝ) (lit_count ps_count : ℕ),
    price_relation a ps_price ∧
    book_count_relation a lit_count ps_count ∧
    a = 20 ∧ ps_price = 24 ∧
    (∀ (x : ℕ), min_cost_condition x → total_cost x ≥ 21600) ∧
    (total_cost 600 = 21600) :=
by
  sorry

end NUMINAMATH_GPT_book_pricing_and_min_cost_l1528_152841


namespace NUMINAMATH_GPT_problem_statement_l1528_152863

noncomputable def f (x a : ℝ) : ℝ := x^2 + (2 * a - 8) * x

theorem problem_statement
  (f : ℝ → ℝ → ℝ)
  (sol_set : Set ℝ)
  (cond1 : ∀ a : ℝ, sol_set = {x : ℝ | -1 ≤ x ∧ x ≤ 5} → ∀ x : ℝ, f x a ≤ 5 ↔ x ∈ sol_set)
  (cond2 : ∀ x : ℝ, ∀ m : ℝ, f x 2 ≥ m^2 - 4 * m - 9) :
  (∃ a : ℝ, a = 2) ∧ (∀ m : ℝ, -1 ≤ m ∧ m ≤ 5) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1528_152863


namespace NUMINAMATH_GPT_algebraic_identity_l1528_152885

theorem algebraic_identity (a b : ℕ) (h1 : a = 753) (h2 : b = 247)
  (identity : ∀ a b, (a^2 + b^2 - a * b) / (a^3 + b^3) = 1 / (a + b)) : 
  (753^2 + 247^2 - 753 * 247) / (753^3 + 247^3) = 0.001 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_identity_l1528_152885


namespace NUMINAMATH_GPT_find_xy_l1528_152852

theorem find_xy (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : (x - y)^2 = 9) : x * y = 3 :=
sorry

end NUMINAMATH_GPT_find_xy_l1528_152852


namespace NUMINAMATH_GPT_relationship_of_sets_l1528_152820

def set_A : Set ℝ := {x | ∃ (k : ℤ), x = (k : ℝ) / 6 + 1}
def set_B : Set ℝ := {x | ∃ (k : ℤ), x = (k : ℝ) / 3 + 1 / 2}
def set_C : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k : ℝ) / 3 + 1 / 2}

theorem relationship_of_sets : set_C ⊆ set_B ∧ set_B ⊆ set_A := by
  sorry

end NUMINAMATH_GPT_relationship_of_sets_l1528_152820


namespace NUMINAMATH_GPT_function_relationship_area_60_maximum_area_l1528_152853

-- Definitions and conditions
def perimeter := 32
def side_length (x : ℝ) : ℝ := 16 - x  -- One side of the rectangle
def area (x : ℝ) : ℝ := x * (16 - x)

-- Theorem 1: Function relationship between y and x
theorem function_relationship (x : ℝ) (hx : 0 < x ∧ x < 16) : area x = -x^2 + 16 * x :=
by
  sorry

-- Theorem 2: Values of x when the area is 60 square meters
theorem area_60 (x : ℝ) (hx1 : area x = 60) : x = 6 ∨ x = 10 :=
by
  sorry

-- Theorem 3: Maximum area
theorem maximum_area : ∃ x, area x = 64 ∧ x = 8 :=
by
  sorry

end NUMINAMATH_GPT_function_relationship_area_60_maximum_area_l1528_152853


namespace NUMINAMATH_GPT_solution_set_inequality_l1528_152840

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) / (x + 2) > 0 ↔ x > 1 ∨ x < -2 :=
by {
  sorry -- proof omitted
}

end NUMINAMATH_GPT_solution_set_inequality_l1528_152840


namespace NUMINAMATH_GPT_sale_record_is_negative_five_l1528_152811

-- Given that a purchase of 10 items is recorded as +10
def purchase_record (items : Int) : Int := items

-- Prove that the sale of 5 items should be recorded as -5
theorem sale_record_is_negative_five : purchase_record 10 = 10 → purchase_record (-5) = -5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sale_record_is_negative_five_l1528_152811


namespace NUMINAMATH_GPT_sum_p_q_eq_21_l1528_152847

theorem sum_p_q_eq_21 (p q : ℤ) :
  {x | x^2 + 6 * x - q = 0} ∩ {x | x^2 - p * x + 6 = 0} = {2} → p + q = 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_p_q_eq_21_l1528_152847


namespace NUMINAMATH_GPT_john_speed_above_limit_l1528_152815

theorem john_speed_above_limit (distance : ℝ) (time : ℝ) (speed_limit : ℝ) 
  (h1 : distance = 150) (h2 : time = 2) (h3 : speed_limit = 60) : 
  (distance / time) - speed_limit = 15 :=
by
  -- steps to show the proof
  sorry

end NUMINAMATH_GPT_john_speed_above_limit_l1528_152815


namespace NUMINAMATH_GPT_faye_age_l1528_152842

variable (C D E F : ℕ)

-- Conditions
axiom h1 : D = 16
axiom h2 : D = E - 4
axiom h3 : E = C + 5
axiom h4 : F = C + 2

-- Goal: Prove that F = 17
theorem faye_age : F = 17 :=
by
  sorry

end NUMINAMATH_GPT_faye_age_l1528_152842


namespace NUMINAMATH_GPT_profit_ratio_l1528_152873

noncomputable def effective_capital (investment : ℕ) (months : ℕ) : ℕ := investment * months

theorem profit_ratio : 
  let P_investment := 4000
  let P_months := 12
  let Q_investment := 9000
  let Q_months := 8
  let P_effective := effective_capital P_investment P_months
  let Q_effective := effective_capital Q_investment Q_months
  (P_effective / Nat.gcd P_effective Q_effective) = 2 ∧ (Q_effective / Nat.gcd P_effective Q_effective) = 3 :=
sorry

end NUMINAMATH_GPT_profit_ratio_l1528_152873


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l1528_152828

theorem ratio_of_x_to_y (x y : ℚ) (h : (2 * x - 3 * y) / (x + 2 * y) = 5 / 4) : x / y = 22 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l1528_152828


namespace NUMINAMATH_GPT_area_of_union_of_triangle_and_reflection_l1528_152846

-- Define points in ℝ²
structure Point where
  x : ℝ
  y : ℝ

-- Define the vertices of the original triangle
def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -1⟩
def C : Point := ⟨7, 0⟩

-- Define the vertices of the reflected triangle
def A' : Point := ⟨-2, 3⟩
def B' : Point := ⟨-4, -1⟩
def C' : Point := ⟨-7, 0⟩

-- Calculate the area of a triangle given three points
def triangleArea (P Q R : Point) : ℝ :=
  0.5 * |P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y)|

-- Statement to prove: the area of the union of the original and reflected triangles
theorem area_of_union_of_triangle_and_reflection :
  triangleArea A B C + triangleArea A' B' C' = 14 := 
sorry

end NUMINAMATH_GPT_area_of_union_of_triangle_and_reflection_l1528_152846


namespace NUMINAMATH_GPT_tank_capacity_l1528_152868

theorem tank_capacity (c w : ℝ) 
  (h1 : w / c = 1 / 7) 
  (h2 : (w + 5) / c = 1 / 5) : 
  c = 87.5 := 
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l1528_152868


namespace NUMINAMATH_GPT_grandpa_age_times_jungmin_age_l1528_152848

-- Definitions based on the conditions
def grandpa_age_last_year : ℕ := 71
def jungmin_age_last_year : ℕ := 8
def grandpa_age_this_year : ℕ := grandpa_age_last_year + 1
def jungmin_age_this_year : ℕ := jungmin_age_last_year + 1

-- The statement to prove
theorem grandpa_age_times_jungmin_age :
  grandpa_age_this_year / jungmin_age_this_year = 8 :=
by
  sorry

end NUMINAMATH_GPT_grandpa_age_times_jungmin_age_l1528_152848


namespace NUMINAMATH_GPT_ratio_of_area_to_square_of_perimeter_of_equilateral_triangle_l1528_152859

theorem ratio_of_area_to_square_of_perimeter_of_equilateral_triangle :
  let a := 10
  let area := (10 * 10 * (Real.sqrt 3) / 4)
  let perimeter := 3 * 10
  let square_of_perimeter := perimeter * perimeter
  (area / square_of_perimeter) = (Real.sqrt 3 / 36) := by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_ratio_of_area_to_square_of_perimeter_of_equilateral_triangle_l1528_152859


namespace NUMINAMATH_GPT_team_A_wins_series_4_1_probability_l1528_152830

noncomputable def probability_team_A_wins_series_4_1 : ℝ :=
  let home_win_prob : ℝ := 0.6
  let away_win_prob : ℝ := 0.5
  let home_loss_prob : ℝ := 1 - home_win_prob
  let away_loss_prob : ℝ := 1 - away_win_prob
  -- Scenario 1: L W W W W
  let p1 := home_loss_prob * home_win_prob * away_win_prob * away_win_prob * home_win_prob
  -- Scenario 2: W L W W W
  let p2 := home_win_prob * home_loss_prob * away_win_prob * away_win_prob * home_win_prob
  -- Scenario 3: W W L W W
  let p3 := home_win_prob * home_win_prob * away_loss_prob * away_win_prob * home_win_prob
  -- Scenario 4: W W W L W
  let p4 := home_win_prob * home_win_prob * away_win_prob * away_loss_prob * home_win_prob
  p1 + p2 + p3 + p4

theorem team_A_wins_series_4_1_probability : 
  probability_team_A_wins_series_4_1 = 0.18 :=
by
  -- This where the proof would go
  sorry

end NUMINAMATH_GPT_team_A_wins_series_4_1_probability_l1528_152830


namespace NUMINAMATH_GPT_maxim_birth_probability_l1528_152870

open Nat

def interval_days (start_date end_date : ℕ) : ℕ :=
  end_date - start_date + 1

def total_days_2007_2008 : ℕ :=
  interval_days 245 2735 -- total days from Sep 2, 2007, to Aug 31, 2008

def days_in_2008 : ℕ :=
  interval_days 305 548  -- total days from Jan 1, 2008, to Aug 31, 2008

def probability_born_in_2008 : ℚ :=
  (days_in_2008 : ℚ) / (total_days_2007_2008 : ℚ)

theorem maxim_birth_probability: probability_born_in_2008 = 244 / 365 := 
  sorry

end NUMINAMATH_GPT_maxim_birth_probability_l1528_152870


namespace NUMINAMATH_GPT_sum_one_to_twenty_nine_l1528_152844

theorem sum_one_to_twenty_nine : (29 / 2) * (1 + 29) = 435 := by
  -- proof
  sorry

end NUMINAMATH_GPT_sum_one_to_twenty_nine_l1528_152844


namespace NUMINAMATH_GPT_most_stable_performance_l1528_152883

-- Given variances for the students' scores
def variance_A : ℝ := 2.1
def variance_B : ℝ := 3.5
def variance_C : ℝ := 9
def variance_D : ℝ := 0.7

-- Prove that student D has the most stable performance
theorem most_stable_performance : 
  variance_D < variance_A ∧ variance_D < variance_B ∧ variance_D < variance_C := 
  by 
    sorry

end NUMINAMATH_GPT_most_stable_performance_l1528_152883


namespace NUMINAMATH_GPT_reciprocal_of_neg3_l1528_152801

theorem reciprocal_of_neg3 : (1 : ℚ) / (-3 : ℚ) = -1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg3_l1528_152801


namespace NUMINAMATH_GPT_number_of_herrings_l1528_152886

theorem number_of_herrings (total_fishes pikes sturgeons herrings : ℕ)
  (h1 : total_fishes = 145)
  (h2 : pikes = 30)
  (h3 : sturgeons = 40)
  (h4 : total_fishes = pikes + sturgeons + herrings) :
  herrings = 75 :=
by
  sorry

end NUMINAMATH_GPT_number_of_herrings_l1528_152886


namespace NUMINAMATH_GPT_parabola_c_value_l1528_152802

theorem parabola_c_value (b c : ℝ) 
  (h1 : 2 * b + c = 6) 
  (h2 : -2 * b + c = 2)
  (vertex_cond : ∃ x y : ℝ, y = x^2 + b * x + c ∧ y = -x + 4) : 
  c = 4 :=
sorry

end NUMINAMATH_GPT_parabola_c_value_l1528_152802


namespace NUMINAMATH_GPT_units_digit_product_first_four_composite_numbers_l1528_152836

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_product_first_four_composite_numbers_l1528_152836


namespace NUMINAMATH_GPT_vector_dot_product_parallel_l1528_152824

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (m : ℝ) : ℝ × ℝ := (m, -1)
noncomputable def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v

theorem vector_dot_product_parallel (m : ℝ) (h_parallel : is_parallel a (a.1 + m, a.2 + (-1))) :
  (a.1 * m + a.2 * (-1) = -5 / 2) :=
sorry

end NUMINAMATH_GPT_vector_dot_product_parallel_l1528_152824


namespace NUMINAMATH_GPT_journey_time_l1528_152827

theorem journey_time
  (t_1 t_2 : ℝ)
  (h1 : t_1 + t_2 = 5)
  (h2 : 40 * t_1 + 60 * t_2 = 240) :
  t_1 = 3 :=
sorry

end NUMINAMATH_GPT_journey_time_l1528_152827


namespace NUMINAMATH_GPT_average_running_time_l1528_152843

variable (s : ℕ) -- Number of seventh graders

-- let sixth graders run 20 minutes per day
-- let seventh graders run 18 minutes per day
-- let eighth graders run 15 minutes per day
-- sixth graders = 3 * seventh graders
-- eighth graders = 2 * seventh graders

def sixthGradersRunningTime : ℕ := 20 * (3 * s)
def seventhGradersRunningTime : ℕ := 18 * s
def eighthGradersRunningTime : ℕ := 15 * (2 * s)

def totalRunningTime : ℕ := sixthGradersRunningTime s + seventhGradersRunningTime s + eighthGradersRunningTime s
def totalStudents : ℕ := 3 * s + s + 2 * s

theorem average_running_time : totalRunningTime s / totalStudents s = 18 :=
by sorry

end NUMINAMATH_GPT_average_running_time_l1528_152843


namespace NUMINAMATH_GPT_train_car_count_l1528_152823

theorem train_car_count
    (cars_first_15_sec : ℕ)
    (time_first_15_sec : ℕ)
    (total_time_minutes : ℕ)
    (total_additional_seconds : ℕ)
    (constant_speed : Prop)
    (h1 : cars_first_15_sec = 9)
    (h2 : time_first_15_sec = 15)
    (h3 : total_time_minutes = 3)
    (h4 : total_additional_seconds = 30)
    (h5 : constant_speed) :
    0.6 * (3 * 60 + 30) = 126 := by
  sorry

end NUMINAMATH_GPT_train_car_count_l1528_152823


namespace NUMINAMATH_GPT_twenty_three_percent_of_number_is_forty_six_l1528_152871

theorem twenty_three_percent_of_number_is_forty_six (x : ℝ) (h : (23 / 100) * x = 46) : x = 200 :=
sorry

end NUMINAMATH_GPT_twenty_three_percent_of_number_is_forty_six_l1528_152871


namespace NUMINAMATH_GPT_probability_king_of_diamonds_top_two_l1528_152800

-- Definitions based on the conditions
def total_cards : ℕ := 54
def king_of_diamonds : ℕ := 1
def jokers : ℕ := 2

-- The main theorem statement proving the probability
theorem probability_king_of_diamonds_top_two :
  let prob := (king_of_diamonds / total_cards) + ((total_cards - 1) / total_cards * king_of_diamonds / (total_cards - 1))
  prob = 1 / 27 :=
by
  sorry

end NUMINAMATH_GPT_probability_king_of_diamonds_top_two_l1528_152800


namespace NUMINAMATH_GPT_symmetric_about_y_l1528_152806

theorem symmetric_about_y (m n : ℤ) (h1 : 2 * n - m = -14) (h2 : m = 4) : (m + n) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_GPT_symmetric_about_y_l1528_152806


namespace NUMINAMATH_GPT_iodine_solution_problem_l1528_152884

theorem iodine_solution_problem (init_concentration : Option ℝ) (init_volume : ℝ)
  (final_concentration : ℝ) (added_volume : ℝ) : 
  init_concentration = none 
  → ∃ x : ℝ, init_volume + added_volume = x :=
by
  sorry

end NUMINAMATH_GPT_iodine_solution_problem_l1528_152884


namespace NUMINAMATH_GPT_monotone_increasing_intervals_exists_x0_implies_p_l1528_152813

noncomputable def f (x : ℝ) := 6 * Real.log x + x ^ 2 - 8 * x
noncomputable def g (x : ℝ) (p : ℝ) := p / x + x ^ 2

theorem monotone_increasing_intervals :
  (∀ x, (0 < x ∧ x ≤ 1) → ∃ ε > 0, ∀ y, x < y → f y > f x) ∧
  (∀ x, (3 ≤ x) → ∃ ε > 0, ∀ y, x < y → f y > f x) := by
  sorry

theorem exists_x0_implies_p :
  (∃ x0, 1 ≤ x0 ∧ x0 ≤ Real.exp 1 ∧ f x0 > g x0 p) → p < -8 := by
  sorry

end NUMINAMATH_GPT_monotone_increasing_intervals_exists_x0_implies_p_l1528_152813


namespace NUMINAMATH_GPT_categorize_numbers_l1528_152803

def numbers : Set (Rat) := {-16, 0.04, 1/2, -2/3, 25, 0, -3.6, -0.3, 4/3}

def is_integer (x : Rat) : Prop := ∃ z : Int, x = z
def is_fraction (x : Rat) : Prop := ∃ (p q : Int), q ≠ 0 ∧ x = p / q
def is_negative (x : Rat) : Prop := x < 0

def integers (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_integer x}
def fractions (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_fraction x}
def negative_rationals (s : Set Rat) : Set Rat := {x | x ∈ s ∧ is_fraction x ∧ is_negative x}

theorem categorize_numbers :
  integers numbers = {-16, 25, 0} ∧
  fractions numbers = {0.04, 1/2, -2/33, -3.6, -0.3, 4/3} ∧
  negative_rationals numbers = {-16, -2/3, -3.6, -0.3} :=
  sorry

end NUMINAMATH_GPT_categorize_numbers_l1528_152803


namespace NUMINAMATH_GPT_find_some_number_l1528_152829

theorem find_some_number (x : ℤ) (h : 45 - (28 - (x - (15 - 20))) = 59) : x = 37 :=
by
  sorry

end NUMINAMATH_GPT_find_some_number_l1528_152829


namespace NUMINAMATH_GPT_Gabrielle_sells_8_crates_on_Wednesday_l1528_152845

-- Definitions based on conditions from part a)
def crates_sold_on_Monday := 5
def crates_sold_on_Tuesday := 2 * crates_sold_on_Monday
def crates_sold_on_Thursday := crates_sold_on_Tuesday / 2
def total_crates_sold := 28
def crates_sold_on_Wednesday := total_crates_sold - (crates_sold_on_Monday + crates_sold_on_Tuesday + crates_sold_on_Thursday)

-- The theorem to prove the question == answer given conditions
theorem Gabrielle_sells_8_crates_on_Wednesday : crates_sold_on_Wednesday = 8 := by
  sorry

end NUMINAMATH_GPT_Gabrielle_sells_8_crates_on_Wednesday_l1528_152845


namespace NUMINAMATH_GPT_find_k_l1528_152851

theorem find_k (k : ℚ) : (∀ x y : ℚ, (x, y) = (2, 1) → 3 * k * x - k = -4 * y - 2) → k = -(6 / 5) :=
by
  intro h
  have key := h 2 1 rfl
  have : 3 * k * 2 - k = -4 * 1 - 2 := key
  linarith

end NUMINAMATH_GPT_find_k_l1528_152851


namespace NUMINAMATH_GPT_max_det_value_l1528_152821

theorem max_det_value :
  ∃ θ : ℝ, 
    (1 * ((5 + Real.sin θ) * 9 - 6 * 8) 
     - 2 * (4 * 9 - 6 * (7 + Real.cos θ)) 
     + 3 * (4 * 8 - (5 + Real.sin θ) * (7 + Real.cos θ))) 
     = 93 :=
sorry

end NUMINAMATH_GPT_max_det_value_l1528_152821


namespace NUMINAMATH_GPT_decimal_to_fraction_l1528_152874

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end NUMINAMATH_GPT_decimal_to_fraction_l1528_152874


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l1528_152862

theorem distance_between_parallel_lines (A B C1 C2 : ℝ) (hA : A = 2) (hB : B = 4)
  (hC1 : C1 = -8) (hC2 : C2 = 7) : 
  (|C2 - C1| / (Real.sqrt (A^2 + B^2)) = 3 * Real.sqrt 5 / 2) :=
by
  rw [hA, hB, hC1, hC2]
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l1528_152862


namespace NUMINAMATH_GPT_quintuplets_babies_l1528_152808

theorem quintuplets_babies (t r q : ℕ) (h1 : r = 6 * q)
  (h2 : t = 2 * r)
  (h3 : 2 * t + 3 * r + 5 * q = 1500) :
  5 * q = 160 :=
by
  sorry

end NUMINAMATH_GPT_quintuplets_babies_l1528_152808


namespace NUMINAMATH_GPT_least_integer_value_l1528_152889

theorem least_integer_value (x : ℝ) (h : |3 * x - 4| ≤ 25) : x = -7 :=
sorry

end NUMINAMATH_GPT_least_integer_value_l1528_152889


namespace NUMINAMATH_GPT_units_digit_a2019_l1528_152858

theorem units_digit_a2019 (a : ℕ → ℝ) (h₁ : ∀ n, a n > 0)
  (h₂ : a 2 ^ 2 + a 4 ^ 2 = 900 - 2 * a 1 * a 5)
  (h₃ : a 5 = 9 * a 3) : (3^(2018) % 10) = 9 := by
  sorry

end NUMINAMATH_GPT_units_digit_a2019_l1528_152858


namespace NUMINAMATH_GPT_true_statements_l1528_152810

theorem true_statements :
  (5 ∣ 25) ∧ (19 ∣ 209 ∧ ¬ (19 ∣ 63)) ∧ (30 ∣ 90) ∧ (14 ∣ 28 ∧ 14 ∣ 56) ∧ (9 ∣ 180) :=
by
  have A : 5 ∣ 25 := sorry
  have B1 : 19 ∣ 209 := sorry
  have B2 : ¬ (19 ∣ 63) := sorry
  have C : 30 ∣ 90 := sorry
  have D1 : 14 ∣ 28 := sorry
  have D2 : 14 ∣ 56 := sorry
  have E : 9 ∣ 180 := sorry
  exact ⟨A, ⟨B1, B2⟩, C, ⟨D1, D2⟩, E⟩

end NUMINAMATH_GPT_true_statements_l1528_152810


namespace NUMINAMATH_GPT_evaluate_expr_correct_l1528_152834

def evaluate_expr : Prop :=
  (8 : ℝ) / (4 * 25) = (0.8 : ℝ) / (0.4 * 25)

theorem evaluate_expr_correct : evaluate_expr :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expr_correct_l1528_152834


namespace NUMINAMATH_GPT_prob_divisible_by_5_of_digits_ending_in_7_l1528_152805

theorem prob_divisible_by_5_of_digits_ending_in_7 :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000 ∧ N % 10 = 7) → (0 : ℚ) = 0 :=
by
  intro N
  sorry

end NUMINAMATH_GPT_prob_divisible_by_5_of_digits_ending_in_7_l1528_152805


namespace NUMINAMATH_GPT_not_snowing_next_five_days_l1528_152857

-- Define the given condition
def prob_snow : ℚ := 2 / 3

-- Define the question condition regarding not snowing for one day
def prob_no_snow : ℚ := 1 - prob_snow

-- Define the question asking for not snowing over 5 days and the expected probability
def prob_no_snow_five_days : ℚ := prob_no_snow ^ 5

theorem not_snowing_next_five_days :
  prob_no_snow_five_days = 1 / 243 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_not_snowing_next_five_days_l1528_152857


namespace NUMINAMATH_GPT_don_walking_speed_l1528_152812

theorem don_walking_speed 
  (distance_between_homes : ℝ)
  (cara_walking_speed : ℝ)
  (cara_distance_before_meeting : ℝ)
  (time_don_starts_after_cara : ℝ)
  (total_distance : distance_between_homes = 45)
  (cara_speed : cara_walking_speed = 6)
  (cara_distance : cara_distance_before_meeting = 30)
  (time_after_cara : time_don_starts_after_cara = 2) :
  ∃ (v : ℝ), v = 5 := by
    sorry

end NUMINAMATH_GPT_don_walking_speed_l1528_152812


namespace NUMINAMATH_GPT_manufacturing_employees_percentage_l1528_152894

theorem manufacturing_employees_percentage 
  (total_circle_deg : ℝ := 360) 
  (manufacturing_deg : ℝ := 18) 
  (sector_proportion : ∀ x y, x / y = (x/y : ℝ)) 
  (percentage : ∀ x, x * 100 = (x * 100 : ℝ)) :
  (manufacturing_deg / total_circle_deg) * 100 = 5 := 
by sorry

end NUMINAMATH_GPT_manufacturing_employees_percentage_l1528_152894
