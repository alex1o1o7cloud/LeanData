import Mathlib

namespace log_base_a1_13_l226_22654

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem log_base_a1_13 (a : ℕ → ℝ) :
  geometric_sequence a → a 9 = 13 → a 13 = 1 → Real.log 13 / Real.log (a 1) = 1 / 3 := by
  sorry


end log_base_a1_13_l226_22654


namespace town_street_lights_l226_22666

/-- Calculates the total number of street lights in a town given the number of neighborhoods,
    roads per neighborhood, and street lights per side of each road. -/
def totalStreetLights (neighborhoods : ℕ) (roadsPerNeighborhood : ℕ) (lightsPerSide : ℕ) : ℕ :=
  neighborhoods * roadsPerNeighborhood * lightsPerSide * 2

/-- Theorem stating that the total number of street lights in the described town is 20000. -/
theorem town_street_lights :
  totalStreetLights 10 4 250 = 20000 := by
  sorry

end town_street_lights_l226_22666


namespace deposit_calculation_l226_22610

theorem deposit_calculation (initial_deposit : ℚ) : 
  (initial_deposit - initial_deposit / 4 - (initial_deposit - initial_deposit / 4) * 4 / 9 - 640) = 3 / 20 * initial_deposit →
  initial_deposit = 2400 := by
sorry

end deposit_calculation_l226_22610


namespace min_disks_needed_prove_min_disks_l226_22651

/-- Represents the storage capacity of a disk in MB -/
def disk_capacity : ℚ := 2

/-- Represents the total number of files -/
def total_files : ℕ := 36

/-- Represents the number of 1.2 MB files -/
def large_files : ℕ := 5

/-- Represents the number of 0.6 MB files -/
def medium_files : ℕ := 16

/-- Represents the size of large files in MB -/
def large_file_size : ℚ := 1.2

/-- Represents the size of medium files in MB -/
def medium_file_size : ℚ := 0.6

/-- Represents the size of small files in MB -/
def small_file_size : ℚ := 0.2

/-- Calculates the number of small files -/
def small_files : ℕ := total_files - large_files - medium_files

/-- Theorem stating the minimum number of disks needed -/
theorem min_disks_needed : ℕ := 14

/-- Proof of the minimum number of disks needed -/
theorem prove_min_disks : min_disks_needed = 14 := by
  sorry

end min_disks_needed_prove_min_disks_l226_22651


namespace quadratic_properties_l226_22615

variable (a b c p q : ℝ)
variable (f : ℝ → ℝ)

-- Define the quadratic function
def is_quadratic (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_properties (h : is_quadratic f a b c) (hpq : p ≠ q) :
  (f p = f q → f (p + q) = c) ∧
  (f (p + q) = c → p + q = 0 ∨ f p = f q) := by
  sorry

end quadratic_properties_l226_22615


namespace arithmetic_sequence_sum_mod_l226_22608

theorem arithmetic_sequence_sum_mod (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 3 →
  d = 5 →
  aₙ = 103 →
  0 ≤ n →
  n < 17 →
  (n : ℤ) ≡ (n * (a₁ + aₙ) / 2) [ZMOD 17] →
  n = 8 :=
sorry

end arithmetic_sequence_sum_mod_l226_22608


namespace fraction_equality_l226_22652

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 25)
  (h2 : c / b = 5)
  (h3 : c / d = 1 / 8) :
  a / d = 5 / 8 := by
  sorry

end fraction_equality_l226_22652


namespace unique_integer_congruence_l226_22659

theorem unique_integer_congruence : ∃! n : ℤ, 4 ≤ n ∧ n ≤ 8 ∧ n ≡ 7882 [ZMOD 5] := by
  sorry

end unique_integer_congruence_l226_22659


namespace sally_boxes_proof_l226_22648

/-- The number of boxes Sally sold on Saturday -/
def saturday_boxes : ℕ := 60

/-- The number of boxes Sally sold on Sunday -/
def sunday_boxes : ℕ := (3 * saturday_boxes) / 2

/-- The total number of boxes Sally sold over two days -/
def total_boxes : ℕ := 150

theorem sally_boxes_proof :
  saturday_boxes + sunday_boxes = total_boxes ∧
  sunday_boxes = (3 * saturday_boxes) / 2 :=
sorry

end sally_boxes_proof_l226_22648


namespace certain_number_proof_l226_22673

theorem certain_number_proof : ∃ x : ℝ, (7.5 * 7.5) + x + (2.5 * 2.5) = 100 :=
by
  sorry

end certain_number_proof_l226_22673


namespace company_employees_l226_22679

/-- 
Given a company that had 15% more employees in December than in January,
and 460 employees in December, prove that it had 400 employees in January.
-/
theorem company_employees (december_employees : ℕ) (january_employees : ℕ) : 
  december_employees = 460 ∧ 
  december_employees = january_employees + (january_employees * 15 / 100) →
  january_employees = 400 := by
sorry

end company_employees_l226_22679


namespace girls_distance_calculation_l226_22688

/-- The number of laps run by boys -/
def boys_laps : ℕ := 124

/-- The additional laps run by girls compared to boys -/
def extra_girls_laps : ℕ := 48

/-- The fraction of a mile per lap -/
def mile_per_lap : ℚ := 5 / 13

/-- The distance run by girls in miles -/
def girls_distance : ℚ := (boys_laps + extra_girls_laps) * mile_per_lap

theorem girls_distance_calculation :
  girls_distance = (124 + 48) * (5 / 13) := by sorry

end girls_distance_calculation_l226_22688


namespace product_divisible_by_seven_l226_22695

theorem product_divisible_by_seven :
  (7 * 17 * 27 * 37 * 47 * 57 * 67) % 7 = 0 := by
  sorry

end product_divisible_by_seven_l226_22695


namespace exists_21_game_period_l226_22609

-- Define the type for the sequence of cumulative games
def CumulativeGames := Nat → Nat

-- Define the properties of the sequence
def ValidSequence (a : CumulativeGames) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, a (n + 7) - a n ≤ 10) ∧
  (a 0 ≥ 1) ∧ (a 42 ≤ 60)

-- Theorem statement
theorem exists_21_game_period (a : CumulativeGames) 
  (h : ValidSequence a) : 
  ∃ k n : Nat, k + n ≤ 42 ∧ a (k + n) - a k = 21 := by
  sorry

end exists_21_game_period_l226_22609


namespace complex_perpendicular_l226_22649

theorem complex_perpendicular (z₁ z₂ : ℂ) (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0) :
  Complex.abs (z₁ + z₂) = Complex.abs (z₁ - z₂) → z₁.re * z₂.re + z₁.im * z₂.im = 0 :=
by sorry

end complex_perpendicular_l226_22649


namespace sunshine_orchard_pumpkins_l226_22640

/-- The number of pumpkins at Moonglow Orchard -/
def x : ℕ := 14

/-- The number of pumpkins at Sunshine Orchard -/
def y : ℕ := 3 * x^2 + 12

theorem sunshine_orchard_pumpkins : y = 600 := by
  sorry

end sunshine_orchard_pumpkins_l226_22640


namespace work_rates_solution_l226_22602

/-- Work rates of workers -/
structure WorkRates where
  casey : ℚ
  bill : ℚ
  alec : ℚ

/-- Given conditions about job completion times -/
def job_conditions (w : WorkRates) : Prop :=
  10 * (w.casey + w.bill) = 1 ∧
  9 * (w.casey + w.alec) = 1 ∧
  8 * (w.alec + w.bill) = 1

/-- Theorem stating the work rates of Casey, Bill, and Alec -/
theorem work_rates_solution :
  ∃ w : WorkRates,
    job_conditions w ∧
    w.casey = (12.8 - 41) / 720 ∧
    w.bill = 41 / 720 ∧
    w.alec = 49 / 720 := by
  sorry

end work_rates_solution_l226_22602


namespace system_solution_l226_22655

/-- Prove that the solution to the system of equations:
    4x - 3y = -10
    6x + 5y = -13
    is (-89/38, 0.21053) -/
theorem system_solution : 
  ∃ (x y : ℝ), 
    (4 * x - 3 * y = -10) ∧ 
    (6 * x + 5 * y = -13) ∧ 
    (x = -89 / 38) ∧ 
    (y = 0.21053) := by
  sorry

end system_solution_l226_22655


namespace cube_root_of_a_plus_b_l226_22672

theorem cube_root_of_a_plus_b (a b : ℝ) (ha : a > 0) 
  (h1 : (2*b - 1)^2 = a) (h2 : (b + 4)^2 = a) (h3 : (2*b - 1) + (b + 4) = 0) : 
  (a + b)^(1/3 : ℝ) = 2 := by
  sorry

end cube_root_of_a_plus_b_l226_22672


namespace junk_mail_distribution_l226_22650

/-- The number of blocks in the neighborhood -/
def num_blocks : ℕ := 16

/-- The number of junk mail pieces given to each house -/
def mail_per_house : ℕ := 4

/-- The total number of junk mail pieces given out -/
def total_mail : ℕ := 1088

/-- The number of houses in each block -/
def houses_per_block : ℕ := 17

theorem junk_mail_distribution :
  houses_per_block * num_blocks * mail_per_house = total_mail :=
by sorry

end junk_mail_distribution_l226_22650


namespace solution_set_abs_fraction_l226_22661

theorem solution_set_abs_fraction (x : ℝ) : 
  (|x / (x - 1)| = x / (x - 1)) ↔ (x ≤ 0 ∨ x > 1) :=
sorry

end solution_set_abs_fraction_l226_22661


namespace k_value_proof_l226_22643

theorem k_value_proof (k : ℝ) (h1 : k ≠ 0) 
  (h2 : ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 12)) : 
  k = 12 := by
  sorry

end k_value_proof_l226_22643


namespace quadratic_intersection_l226_22638

/-- Given two quadratic functions f(x) = ax^2 + bx + c and g(x) = 4ax^2 + 2bx + c,
    where b ≠ 0 and c ≠ 0, their intersection points are x = 0 and x = -b/(3a) -/
theorem quadratic_intersection
  (a b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f := fun x : ℝ => a * x^2 + b * x + c
  let g := fun x : ℝ => 4 * a * x^2 + 2 * b * x + c
  (∃ y, f 0 = y ∧ g 0 = y) ∧
  (∃ y, f (-b / (3 * a)) = y ∧ g (-b / (3 * a)) = y) :=
by sorry

end quadratic_intersection_l226_22638


namespace f_maximum_properties_l226_22681

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem f_maximum_properties (x₀ : ℝ) 
  (h1 : ∀ x > 0, f x ≤ f x₀) 
  (h2 : x₀ > 0) : 
  f x₀ = x₀ ∧ f x₀ > 1/9 := by
  sorry

end f_maximum_properties_l226_22681


namespace simplify_trig_expression_l226_22653

theorem simplify_trig_expression (α : Real) (h : α ∈ Set.Ioo (π/2) (3*π/4)) :
  Real.sqrt (2 - 2 * Real.sin (2 * α)) - Real.sqrt (1 + Real.cos (2 * α)) = Real.sqrt 2 * Real.sin α := by
  sorry

end simplify_trig_expression_l226_22653


namespace correct_systematic_sample_l226_22635

def systematicSample (totalItems : Nat) (sampleSize : Nat) : List Nat :=
  sorry

theorem correct_systematic_sample :
  let totalItems : Nat := 50
  let sampleSize : Nat := 5
  let samplingInterval : Nat := totalItems / sampleSize
  let sample := systematicSample totalItems sampleSize
  samplingInterval = 10 ∧ sample = [7, 17, 27, 37, 47] := by sorry

end correct_systematic_sample_l226_22635


namespace moe_has_least_money_l226_22645

-- Define the set of people
inductive Person : Type
| Bo : Person
| Coe : Person
| Flo : Person
| Jo : Person
| Moe : Person

-- Define the money function
variable (money : Person → ℝ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q
axiom flo_more_than_jo_bo : money Person.Flo > money Person.Jo ∧ money Person.Flo > money Person.Bo
axiom bo_coe_more_than_moe : money Person.Bo > money Person.Moe ∧ money Person.Coe > money Person.Moe
axiom jo_between_bo_moe : money Person.Jo > money Person.Moe ∧ money Person.Jo < money Person.Bo

-- Define the theorem
theorem moe_has_least_money :
  ∀ (p : Person), p ≠ Person.Moe → money Person.Moe < money p :=
sorry

end moe_has_least_money_l226_22645


namespace certain_number_problem_l226_22657

theorem certain_number_problem (x N : ℤ) (h1 : 3 * x = (N - x) + 18) (h2 : x = 11) : N = 26 := by
  sorry

end certain_number_problem_l226_22657


namespace min_value_of_expression_l226_22684

theorem min_value_of_expression (a b c d : ℝ) 
  (hb : b > 0) (hc : c > 0) (ha : a ≥ 0) (hd : d ≥ 0) 
  (h_sum : b + c ≥ a + d) : 
  (b / (c + d) + c / (a + b)) ≥ Real.sqrt 2 - 1/2 := 
sorry

end min_value_of_expression_l226_22684


namespace quadratic_roots_condition_l226_22683

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ + m = 0 ∧ x₂^2 + 4*x₂ + m = 0) → m ≤ 4 := by
  sorry

end quadratic_roots_condition_l226_22683


namespace function_roots_imply_a_range_l226_22614

theorem function_roots_imply_a_range (a b : ℝ) :
  (∀ x : ℝ, (a * x^2 + b * (x + 1) - 2 = x) → (∃ y z : ℝ, y ≠ z ∧ a * y^2 + b * (y + 1) - 2 = y ∧ a * z^2 + b * (z + 1) - 2 = z)) →
  (0 < a ∧ a < 1) :=
by sorry

end function_roots_imply_a_range_l226_22614


namespace trees_chopped_second_half_proof_l226_22698

def trees_chopped_first_half : ℕ := 200
def trees_planted_per_chopped : ℕ := 3
def total_trees_to_plant : ℕ := 1500

def trees_chopped_second_half : ℕ := 300

theorem trees_chopped_second_half_proof :
  trees_chopped_second_half = 
    (total_trees_to_plant - trees_planted_per_chopped * trees_chopped_first_half) / 
    trees_planted_per_chopped := by
  sorry

end trees_chopped_second_half_proof_l226_22698


namespace iris_blueberries_l226_22632

/-- The number of blueberries Iris picked -/
def blueberries : ℕ := 30

/-- The number of cranberries Iris' sister picked -/
def cranberries : ℕ := 20

/-- The number of raspberries Iris' brother picked -/
def raspberries : ℕ := 10

/-- The fraction of total berries that are rotten -/
def rotten_fraction : ℚ := 1/3

/-- The fraction of fresh berries that need to be kept -/
def kept_fraction : ℚ := 1/2

/-- The number of berries they can sell -/
def sellable_berries : ℕ := 20

theorem iris_blueberries :
  blueberries = 30 ∧
  (1 - rotten_fraction) * (1 - kept_fraction) * (blueberries + cranberries + raspberries : ℚ) = sellable_berries := by
  sorry

end iris_blueberries_l226_22632


namespace f_sum_positive_l226_22623

def f (x : ℝ) := x^3 + x

theorem f_sum_positive (a b : ℝ) (h : a + b > 0) : f a + f b > 0 := by
  sorry

end f_sum_positive_l226_22623


namespace methane_moles_in_reaction_l226_22624

/-- 
Proves that the number of moles of Methane combined is equal to 1, given the conditions of the chemical reaction.
-/
theorem methane_moles_in_reaction (x : ℝ) : 
  (x > 0) →  -- Assuming positive number of moles
  (∃ y : ℝ, y > 0 ∧ x + 4 = y + 4) →  -- Mass balance equation
  (1 : ℝ) / x = (1 : ℝ) / 1 →  -- Stoichiometric ratio
  x = 1 := by
sorry

end methane_moles_in_reaction_l226_22624


namespace count_ordered_pairs_l226_22674

theorem count_ordered_pairs (n : ℕ) (hn : n > 1) :
  (Finset.sum (Finset.range (n - 1)) (fun k => n - k)) = (n - 1) * n / 2 := by
  sorry

end count_ordered_pairs_l226_22674


namespace drug_storage_temperature_range_l226_22678

def central_temp : ℝ := 20
def variation : ℝ := 2

def lower_limit : ℝ := central_temp - variation
def upper_limit : ℝ := central_temp + variation

theorem drug_storage_temperature_range : 
  (lower_limit = 18 ∧ upper_limit = 22) := by sorry

end drug_storage_temperature_range_l226_22678


namespace deepak_current_age_l226_22696

/-- Represents the ages of Rahul and Deepak -/
structure Ages where
  rahul : ℕ
  deepak : ℕ

/-- The condition that the ratio of Rahul's age to Deepak's age is 4:3 -/
def ratio_condition (ages : Ages) : Prop :=
  4 * ages.deepak = 3 * ages.rahul

/-- The condition that Rahul will be 26 years old in 6 years -/
def future_condition (ages : Ages) : Prop :=
  ages.rahul + 6 = 26

/-- The theorem stating Deepak's current age given the conditions -/
theorem deepak_current_age (ages : Ages) 
  (h1 : ratio_condition ages) 
  (h2 : future_condition ages) : 
  ages.deepak = 15 := by
  sorry

end deepak_current_age_l226_22696


namespace max_distance_complex_circle_l226_22601

theorem max_distance_complex_circle : 
  ∃ (M : ℝ), M = 7 ∧ 
  ∀ (z : ℂ), Complex.abs (z - (4 - 4*I)) ≤ 2 → Complex.abs (z - 1) ≤ M ∧ 
  ∃ (w : ℂ), Complex.abs (w - (4 - 4*I)) ≤ 2 ∧ Complex.abs (w - 1) = M :=
by sorry

end max_distance_complex_circle_l226_22601


namespace exists_right_triangle_with_different_colors_l226_22629

-- Define the color type
inductive Color
  | Blue
  | Green
  | Red

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the coloring function
def coloring : Point → Color := sorry

-- State the existence of at least one point of each color
axiom exists_blue : ∃ p : Point, coloring p = Color.Blue
axiom exists_green : ∃ p : Point, coloring p = Color.Green
axiom exists_red : ∃ p : Point, coloring p = Color.Red

-- Define a right triangle
def is_right_triangle (p q r : Point) : Prop := sorry

-- State the theorem
theorem exists_right_triangle_with_different_colors :
  ∃ p q r : Point, is_right_triangle p q r ∧
    coloring p ≠ coloring q ∧
    coloring q ≠ coloring r ∧
    coloring r ≠ coloring p :=
sorry

end exists_right_triangle_with_different_colors_l226_22629


namespace kia_vehicles_count_l226_22663

def total_vehicles : ℕ := 400

def dodge_vehicles : ℕ := total_vehicles / 2

def hyundai_vehicles : ℕ := dodge_vehicles / 2

def kia_vehicles : ℕ := total_vehicles - dodge_vehicles - hyundai_vehicles

theorem kia_vehicles_count : kia_vehicles = 100 := by
  sorry

end kia_vehicles_count_l226_22663


namespace translation_right_4_units_l226_22603

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point to the right -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_right_4_units :
  let P : Point := { x := -5, y := 4 }
  let P' : Point := translateRight P 4
  P'.x = -1 ∧ P'.y = 4 := by
  sorry

end translation_right_4_units_l226_22603


namespace umbrella_arrangement_count_l226_22685

/-- The number of ways to arrange n people with distinct heights in an umbrella shape -/
def umbrella_arrangements (n : ℕ) : ℕ :=
  sorry

/-- There are 7 actors with distinct heights to be arranged -/
def num_actors : ℕ := 7

theorem umbrella_arrangement_count :
  umbrella_arrangements num_actors = 20 := by sorry

end umbrella_arrangement_count_l226_22685


namespace inequality_proof_l226_22682

theorem inequality_proof (x : ℝ) : (x^2 - 16) / (x^2 + 10*x + 25) < 0 ↔ -4 < x ∧ x < 4 := by
  sorry

end inequality_proof_l226_22682


namespace student_group_size_l226_22656

/-- The number of students in a group with overlapping class registrations --/
def num_students (history math english all_three two_classes : ℕ) : ℕ :=
  history + math + english - two_classes - 2 * all_three + all_three

theorem student_group_size :
  let history := 19
  let math := 14
  let english := 26
  let all_three := 3
  let two_classes := 7
  num_students history math english all_three two_classes = 46 := by
  sorry

end student_group_size_l226_22656


namespace all_cards_same_number_l226_22605

theorem all_cards_same_number (n : ℕ) (c : Fin n → ℕ) : 
  (∀ i : Fin n, c i ∈ Finset.range n) →
  (∀ s : Finset (Fin n), (s.sum c) % (n + 1) ≠ 0) →
  (∀ i j : Fin n, c i = c j) :=
by sorry

end all_cards_same_number_l226_22605


namespace girls_on_playground_l226_22619

theorem girls_on_playground (total_children boys : ℕ) 
  (h1 : total_children = 117) 
  (h2 : boys = 40) : 
  total_children - boys = 77 := by
sorry

end girls_on_playground_l226_22619


namespace paint_leftover_l226_22669

/-- Given the following conditions:
    1. The total number of paint containers is 16
    2. There are 4 equally-sized walls
    3. One wall is not painted
    4. One container is used for the ceiling
    Prove that the number of leftover paint containers is 3. -/
theorem paint_leftover (total_containers : ℕ) (num_walls : ℕ) (unpainted_walls : ℕ) (ceiling_containers : ℕ) :
  total_containers = 16 →
  num_walls = 4 →
  unpainted_walls = 1 →
  ceiling_containers = 1 →
  total_containers - (num_walls - unpainted_walls) * (total_containers / num_walls) - ceiling_containers = 3 :=
by sorry

end paint_leftover_l226_22669


namespace prop_C_and_D_l226_22618

theorem prop_C_and_D : 
  (∀ a b : ℝ, a > b → a^3 > b^3) ∧ 
  (∀ a b c d : ℝ, (a > b ∧ c > d) → a - d > b - c) := by
  sorry

end prop_C_and_D_l226_22618


namespace parabola_translation_specific_parabola_translation_l226_22620

/-- Represents a parabola in the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a,
    b := p.a * h^2 + p.b + v }

theorem parabola_translation (p : Parabola) (h v : ℝ) :
  (translate p h v).a * (X - h)^2 + (translate p h v).b = p.a * X^2 + p.b + v :=
by sorry

theorem specific_parabola_translation :
  let p : Parabola := { a := 2, b := 3 }
  let translated := translate p 3 2
  translated.a * (X - 3)^2 + translated.b = 2 * (X - 3)^2 + 5 :=
by sorry

end parabola_translation_specific_parabola_translation_l226_22620


namespace sequence_sum_l226_22622

theorem sequence_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) :
  x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 1 →
  4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 12 →
  9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 123 →
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 334 := by
  sorry

end sequence_sum_l226_22622


namespace exam_question_count_l226_22677

theorem exam_question_count :
  ∀ (num_type_a num_type_b : ℕ) (time_per_a time_per_b : ℚ),
    num_type_a = 100 →
    time_per_a = 2 * time_per_b →
    num_type_a * time_per_a = 120 →
    num_type_a * time_per_a + num_type_b * time_per_b = 180 →
    num_type_a + num_type_b = 200 := by
  sorry

end exam_question_count_l226_22677


namespace problem_solution_l226_22671

theorem problem_solution : (π - 3.14) ^ 0 + (-1/2) ^ (-1 : ℤ) + |3 - Real.sqrt 8| - 4 * Real.cos (π/4) = 2 - 4 * Real.sqrt 2 := by
  sorry

end problem_solution_l226_22671


namespace set_equality_l226_22699

theorem set_equality : {x : ℕ | x > 1 ∧ x ≤ 3} = {x : ℕ | x = 2 ∨ x = 3} := by
  sorry

end set_equality_l226_22699


namespace fixed_point_on_line_l226_22616

theorem fixed_point_on_line (m : ℝ) : 
  (m + 2) * (-4/5) + (m - 3) * (4/5) + 4 = 0 := by
  sorry

end fixed_point_on_line_l226_22616


namespace beverage_production_l226_22604

/-- Represents the number of bottles of beverage A -/
def bottles_A : ℕ := sorry

/-- Represents the number of bottles of beverage B -/
def bottles_B : ℕ := sorry

/-- The amount of additive (in grams) required for one bottle of beverage A -/
def additive_A : ℚ := 1/5

/-- The amount of additive (in grams) required for one bottle of beverage B -/
def additive_B : ℚ := 3/10

/-- The total number of bottles produced -/
def total_bottles : ℕ := 200

/-- The total amount of additive used (in grams) -/
def total_additive : ℚ := 54

theorem beverage_production :
  bottles_A + bottles_B = total_bottles ∧
  additive_A * bottles_A + additive_B * bottles_B = total_additive ∧
  bottles_A = 60 ∧
  bottles_B = 140 := by sorry

end beverage_production_l226_22604


namespace cone_volume_ratio_l226_22664

-- Define the ratio of central angles
def angle_ratio : ℚ := 3 / 4

-- Define a function to calculate the volume ratio given the angle ratio
def volume_ratio (r : ℚ) : ℚ := r^2

-- Theorem statement
theorem cone_volume_ratio :
  volume_ratio angle_ratio = 9 / 16 := by
  sorry

end cone_volume_ratio_l226_22664


namespace lcm_48_140_l226_22662

theorem lcm_48_140 : Nat.lcm 48 140 = 1680 := by
  sorry

end lcm_48_140_l226_22662


namespace fixed_point_quadratic_function_l226_22627

theorem fixed_point_quadratic_function (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - (2-m)*x + m
  f (-1) = 3 := by
sorry

end fixed_point_quadratic_function_l226_22627


namespace at_least_four_same_acquaintances_l226_22611

theorem at_least_four_same_acquaintances :
  ∀ (contestants : Finset Nat) (acquaintances : Nat → Finset Nat),
    contestants.card = 90 →
    (∀ x ∈ contestants, (acquaintances x).card ≥ 60) →
    (∀ x ∈ contestants, (acquaintances x) ⊆ contestants) →
    (∀ x ∈ contestants, x ∉ acquaintances x) →
    ∃ n : Nat, ∃ s : Finset Nat, s ⊆ contestants ∧ s.card ≥ 4 ∧
      ∀ x ∈ s, (acquaintances x).card = n :=
by
  sorry


end at_least_four_same_acquaintances_l226_22611


namespace first_car_speed_l226_22689

/-- Proves that the speed of the first car is 40 miles per hour given the conditions of the problem -/
theorem first_car_speed (black_car_speed : ℝ) (initial_distance : ℝ) (overtake_time : ℝ) : 
  black_car_speed = 50 →
  initial_distance = 10 →
  overtake_time = 1 →
  (black_car_speed * overtake_time - initial_distance) / overtake_time = 40 :=
by sorry

end first_car_speed_l226_22689


namespace tank_filling_time_l226_22692

theorem tank_filling_time (fill_rate : ℝ) (leak_rate : ℝ) (fill_time_no_leak : ℝ) (empty_time_leak : ℝ) :
  fill_rate = 1 / fill_time_no_leak →
  leak_rate = 1 / empty_time_leak →
  fill_time_no_leak = 8 →
  empty_time_leak = 72 →
  (1 : ℝ) / (fill_rate - leak_rate) = 9 := by
  sorry

end tank_filling_time_l226_22692


namespace intersection_line_of_circles_l226_22626

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = y
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 = x

-- Define the line
def intersection_line (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ (x y : ℝ), circle_O1 x y ∧ circle_O2 x y → intersection_line x y :=
by sorry

end intersection_line_of_circles_l226_22626


namespace difference_of_squares_l226_22658

theorem difference_of_squares : 49^2 - 25^2 = 1776 := by
  sorry

end difference_of_squares_l226_22658


namespace isosceles_triangle_vertex_angle_l226_22660

theorem isosceles_triangle_vertex_angle (α β γ : ℝ) : 
  -- Triangle conditions
  α + β + γ = 180 ∧
  -- Isosceles triangle condition (two angles are equal)
  (α = β ∨ β = γ ∨ α = γ) ∧
  -- One angle is 80°
  (α = 80 ∨ β = 80 ∨ γ = 80) →
  -- The vertex angle (the one that's not equal to the other two) is either 20° or 80°
  (α ≠ β → γ = 20 ∨ γ = 80) ∧
  (β ≠ γ → α = 20 ∨ α = 80) ∧
  (α ≠ γ → β = 20 ∨ β = 80) :=
by sorry

end isosceles_triangle_vertex_angle_l226_22660


namespace perpendicular_lines_m_values_l226_22621

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∀ x y : ℝ, mx - y + 1 = 0 ∧ x + m^2*y - 2 = 0 → 
   (m * 1) * 1 + 1 * m^2 = 0) → 
  m = 0 ∨ m = 1 := by
sorry

end perpendicular_lines_m_values_l226_22621


namespace cube_root_equation_solution_l226_22668

theorem cube_root_equation_solution (y : ℝ) :
  (5 - 2 / y) ^ (1/3 : ℝ) = -3 → y = 1/16 :=
by sorry

end cube_root_equation_solution_l226_22668


namespace partial_fraction_decomposition_l226_22642

theorem partial_fraction_decomposition :
  ∀ (A B C : ℝ),
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 →
    (x^2 - 7) / ((x - 2) * (x - 3) * (x - 5)) = A / (x - 2) + B / (x - 3) + C / (x - 5)) ↔
  A = -1 ∧ B = -1 ∧ C = 3 :=
by sorry

end partial_fraction_decomposition_l226_22642


namespace hyperbola_slope_theorem_l226_22675

/-- A hyperbola passing through specific points with given asymptote slopes -/
structure Hyperbola where
  -- Points the hyperbola passes through
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ
  point4 : ℝ × ℝ
  -- Slope of one asymptote
  slope1 : ℚ
  -- Slope of the other asymptote
  slope2 : ℚ
  -- Condition that the hyperbola passes through the given points
  passes_through : point1 = (2, 5) ∧ point2 = (7, 3) ∧ point3 = (1, 1) ∧ point4 = (10, 10)
  -- Condition that slope1 is 20/17
  slope1_value : slope1 = 20/17
  -- Condition that the product of slopes is -1
  slopes_product : slope1 * slope2 = -1

theorem hyperbola_slope_theorem (h : Hyperbola) :
  h.slope2 = -17/20 ∧ (100 * 17 + 20 = 1720) := by
  sorry

#check hyperbola_slope_theorem

end hyperbola_slope_theorem_l226_22675


namespace find_A_l226_22647

theorem find_A : ∀ A B : ℕ,
  (A ≥ 1 ∧ A ≤ 9) →
  (B ≥ 0 ∧ B ≤ 9) →
  (10 * A + 3 ≥ 10 ∧ 10 * A + 3 ≤ 99) →
  (610 + B ≥ 100 ∧ 610 + B ≤ 999) →
  (10 * A + 3) + (610 + B) = 695 →
  A = 8 := by
sorry

end find_A_l226_22647


namespace percentage_increase_proof_l226_22600

theorem percentage_increase_proof (original_earnings new_earnings : ℝ) 
  (h1 : original_earnings = 60)
  (h2 : new_earnings = 84) :
  ((new_earnings - original_earnings) / original_earnings) * 100 = 40 := by
sorry

end percentage_increase_proof_l226_22600


namespace infinitely_many_odd_terms_l226_22687

theorem infinitely_many_odd_terms (n : ℕ) (hn : n > 1) :
  ∀ m : ℕ, ∃ k > m, Odd (⌊(n^k : ℝ) / k⌋) := by
  sorry

end infinitely_many_odd_terms_l226_22687


namespace correct_vs_incorrect_calculation_l226_22612

theorem correct_vs_incorrect_calculation : 
  (12 - (3 * 4)) - ((12 - 3) * 4) = -36 := by sorry

end correct_vs_incorrect_calculation_l226_22612


namespace pizzas_successfully_served_l226_22644

theorem pizzas_successfully_served 
  (total_served : ℕ) 
  (returned : ℕ) 
  (h1 : total_served = 9) 
  (h2 : returned = 6) : 
  total_served - returned = 3 :=
by sorry

end pizzas_successfully_served_l226_22644


namespace mike_picked_52_peaches_l226_22625

/-- The number of peaches Mike picked -/
def peaches_picked (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that Mike picked 52 peaches -/
theorem mike_picked_52_peaches : peaches_picked 34 86 = 52 := by
  sorry

end mike_picked_52_peaches_l226_22625


namespace exam_score_calculation_l226_22694

/-- Calculate total marks in an exam with penalties for incorrect answers -/
theorem exam_score_calculation 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (marks_per_correct : ℕ) 
  (penalty_per_wrong : ℕ) :
  total_questions = 60 →
  correct_answers = 36 →
  marks_per_correct = 4 →
  penalty_per_wrong = 1 →
  (correct_answers * marks_per_correct) - 
  ((total_questions - correct_answers) * penalty_per_wrong) = 120 :=
by
  sorry

end exam_score_calculation_l226_22694


namespace smallest_max_sum_l226_22686

theorem smallest_max_sum (a b c d e f : ℕ+) 
  (sum_eq : a + b + c + d + e + f = 2512) : 
  (∃ (M : ℕ), M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) ∧ 
   (∀ (M' : ℕ), M' = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) → M ≤ M') ∧
   M = 1005) := by
  sorry

end smallest_max_sum_l226_22686


namespace john_pushups_l226_22630

theorem john_pushups (zachary_pushups : ℕ) (david_more_than_zachary : ℕ) (john_less_than_david : ℕ)
  (h1 : zachary_pushups = 51)
  (h2 : david_more_than_zachary = 22)
  (h3 : john_less_than_david = 4) :
  zachary_pushups + david_more_than_zachary - john_less_than_david = 69 :=
by
  sorry

end john_pushups_l226_22630


namespace odd_function_value_l226_22693

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x - 3*x + k else -(2^(-x) - 3*(-x) + k)
  where k : ℝ := -1 -- We define k here to make the function complete

-- State the theorem
theorem odd_function_value : f (-1) = 2 := by
  sorry

end odd_function_value_l226_22693


namespace triangle_area_example_l226_22617

/-- The area of a triangle given its vertices -/
def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  let v := (a.1 - c.1, a.2 - c.2)
  let w := (b.1 - c.1, b.2 - c.2)
  0.5 * abs (v.1 * w.2 - v.2 * w.1)

/-- Theorem: The area of the triangle with vertices (3, -5), (-2, 0), and (5, -8) is 2.5 -/
theorem triangle_area_example : triangleArea (3, -5) (-2, 0) (5, -8) = 2.5 := by
  sorry


end triangle_area_example_l226_22617


namespace stock_price_problem_l226_22633

theorem stock_price_problem (price_less_expensive : ℝ) (price_more_expensive : ℝ) : 
  price_more_expensive = 2 * price_less_expensive →
  14 * price_more_expensive + 26 * price_less_expensive = 2106 →
  price_more_expensive = 78 := by
sorry

end stock_price_problem_l226_22633


namespace three_digit_numbers_after_exclusion_l226_22697

/-- The count of three-digit numbers (100 to 999) -/
def total_three_digit_numbers : ℕ := 900

/-- The count of numbers in the form ABA where A and B are digits and A ≠ 0 -/
def count_ABA : ℕ := 81

/-- The count of numbers in the form AAB or BAA where A and B are digits and A ≠ 0 -/
def count_AAB_BAA : ℕ := 81

/-- The total count of excluded numbers -/
def total_excluded : ℕ := count_ABA + count_AAB_BAA

theorem three_digit_numbers_after_exclusion :
  total_three_digit_numbers - total_excluded = 738 := by sorry

end three_digit_numbers_after_exclusion_l226_22697


namespace seth_oranges_l226_22665

theorem seth_oranges (initial_boxes : ℕ) : 
  (initial_boxes - 1) / 2 = 4 → initial_boxes = 9 :=
by
  sorry

end seth_oranges_l226_22665


namespace gain_percent_calculation_l226_22631

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h : 50 * cost_price = 32 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 56.25 := by
  sorry

end gain_percent_calculation_l226_22631


namespace negative_square_root_operations_l226_22670

theorem negative_square_root_operations :
  (-Real.sqrt (2^2) < 0) ∧
  ((Real.sqrt 2)^2 ≥ 0) ∧
  (Real.sqrt (2^2) ≥ 0) ∧
  (Real.sqrt ((-2)^2) ≥ 0) :=
by sorry

end negative_square_root_operations_l226_22670


namespace combined_age_l226_22628

theorem combined_age (tony_age belinda_age : ℕ) : 
  tony_age = 16 →
  belinda_age = 40 →
  belinda_age = 2 * tony_age + 8 →
  tony_age + belinda_age = 56 :=
by sorry

end combined_age_l226_22628


namespace sequence_formula_T_formula_l226_22667

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom S_def (n : ℕ) : n > 0 → S n = 2 * sequence_a n - 2

theorem sequence_formula (n : ℕ) (h : n > 0) : sequence_a n = 2^n := by sorry

def T (n : ℕ) : ℝ := sorry

theorem T_formula (n : ℕ) (h : n > 0) : T n = 2^(n+2) - 4 - 2*n := by sorry

end sequence_formula_T_formula_l226_22667


namespace non_congruent_triangles_count_l226_22680

-- Define the type for 2D points
structure Point where
  x : ℝ
  y : ℝ

-- Define the set of points
def points : List Point := [
  ⟨0, 0⟩, ⟨1, 0⟩, ⟨2, 0⟩,
  ⟨0, 1⟩, ⟨1, 1⟩, ⟨2, 1⟩,
  ⟨0.5, 2⟩, ⟨1.5, 2⟩, ⟨2.5, 2⟩
]

-- Function to check if two triangles are congruent
def are_congruent (t1 t2 : List Point) : Prop := sorry

-- Function to count non-congruent triangles
def count_non_congruent_triangles (pts : List Point) : ℕ := sorry

-- Theorem stating the number of non-congruent triangles
theorem non_congruent_triangles_count :
  count_non_congruent_triangles points = 18 := by sorry

end non_congruent_triangles_count_l226_22680


namespace product_evaluation_l226_22690

theorem product_evaluation (n : ℕ) (h : n = 3) : 
  (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 5040 := by
  sorry

end product_evaluation_l226_22690


namespace sqrt_4_times_9_sqrt_49_over_36_cube_root_a_to_6_sqrt_9a_squared_l226_22607

-- Part a
theorem sqrt_4_times_9 : Real.sqrt (4 * 9) = 6 := by sorry

-- Part b
theorem sqrt_49_over_36 : Real.sqrt (49 / 36) = 7 / 6 := by sorry

-- Part c
theorem cube_root_a_to_6 (a : ℝ) : (a^6)^(1/3 : ℝ) = a^2 := by sorry

-- Part d
theorem sqrt_9a_squared (a : ℝ) : Real.sqrt (9 * a^2) = 3 * a := by sorry

end sqrt_4_times_9_sqrt_49_over_36_cube_root_a_to_6_sqrt_9a_squared_l226_22607


namespace average_glasses_per_box_l226_22691

/-- Prove that the average number of glasses per box is 15, given the following conditions:
  - There are two types of boxes: small (12 glasses) and large (16 glasses)
  - There are 16 more large boxes than small boxes
  - The total number of glasses is 480
-/
theorem average_glasses_per_box (small_box : ℕ) (large_box : ℕ) :
  small_box * 12 + large_box * 16 = 480 →
  large_box = small_box + 16 →
  (480 : ℚ) / (small_box + large_box) = 15 := by
sorry


end average_glasses_per_box_l226_22691


namespace librarian_crates_l226_22641

theorem librarian_crates (novels comics documentaries albums : ℕ) 
  (items_per_crate : ℕ) (h1 : novels = 145) (h2 : comics = 271) 
  (h3 : documentaries = 419) (h4 : albums = 209) (h5 : items_per_crate = 9) : 
  (novels + comics + documentaries + albums + items_per_crate - 1) / items_per_crate = 117 := by
  sorry

end librarian_crates_l226_22641


namespace bellas_score_l226_22637

theorem bellas_score (total_students : ℕ) (avg_without_bella : ℚ) (avg_with_bella : ℚ) :
  total_students = 20 →
  avg_without_bella = 82 →
  avg_with_bella = 85 →
  (total_students * avg_with_bella - (total_students - 1) * avg_without_bella : ℚ) = 142 :=
by sorry

end bellas_score_l226_22637


namespace base_ten_solution_l226_22634

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Checks if the equation 253_b + 146_b = 410_b holds for a given base b --/
def equation_holds (b : Nat) : Prop :=
  to_decimal [2, 5, 3] b + to_decimal [1, 4, 6] b = to_decimal [4, 1, 0] b

theorem base_ten_solution :
  ∃ (b : Nat), b > 9 ∧ equation_holds b ∧ ∀ (x : Nat), x ≠ b → ¬(equation_holds x) :=
sorry

end base_ten_solution_l226_22634


namespace solve_linear_equation_l226_22613

theorem solve_linear_equation (x : ℝ) :
  3 * x - 7 = 11 → x = 6 := by
sorry

end solve_linear_equation_l226_22613


namespace base_conversion_l226_22636

theorem base_conversion (b : ℝ) (h : b > 0) : 53 = 1 * b^2 + 0 * b + 3 → b = Real.sqrt 30 := by
  sorry

end base_conversion_l226_22636


namespace fifth_quiz_score_l226_22606

def quiz_scores : List ℕ := [90, 98, 92, 94]
def desired_average : ℕ := 94
def total_quizzes : ℕ := 5

theorem fifth_quiz_score (scores : List ℕ) (avg : ℕ) (total : ℕ) :
  scores = quiz_scores ∧ avg = desired_average ∧ total = total_quizzes →
  (scores.sum + (avg * total - scores.sum)) / total = avg ∧
  avg * total - scores.sum = 96 := by
  sorry

end fifth_quiz_score_l226_22606


namespace second_platform_length_l226_22646

/-- Given a train and two platforms, calculate the length of the second platform. -/
theorem second_platform_length
  (train_length : ℝ)
  (first_platform_length : ℝ)
  (first_crossing_time : ℝ)
  (second_crossing_time : ℝ)
  (h1 : train_length = 30)
  (h2 : first_platform_length = 90)
  (h3 : first_crossing_time = 12)
  (h4 : second_crossing_time = 15)
  (h5 : train_length > 0)
  (h6 : first_platform_length > 0)
  (h7 : first_crossing_time > 0)
  (h8 : second_crossing_time > 0) :
  let speed := (train_length + first_platform_length) / first_crossing_time
  let second_platform_length := speed * second_crossing_time - train_length
  second_platform_length = 120 := by
sorry


end second_platform_length_l226_22646


namespace inequality_proof_l226_22676

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x*y + y*z + z*x ≤ 1) : 
  (x + 1/x) * (y + 1/y) * (z + 1/z) ≥ 8 * (x + y) * (y + z) * (z + x) := by
  sorry

end inequality_proof_l226_22676


namespace probability_product_less_than_30_l226_22639

def paco_spinner : Finset ℕ := Finset.range 5
def manu_spinner : Finset ℕ := Finset.range 12

def product_less_than_30 (x : ℕ) (y : ℕ) : Bool :=
  x * y < 30

theorem probability_product_less_than_30 :
  (Finset.filter (λ (pair : ℕ × ℕ) => product_less_than_30 (pair.1 + 1) (pair.2 + 1))
    (paco_spinner.product manu_spinner)).card / (paco_spinner.card * manu_spinner.card : ℚ) = 51 / 60 :=
sorry

end probability_product_less_than_30_l226_22639
