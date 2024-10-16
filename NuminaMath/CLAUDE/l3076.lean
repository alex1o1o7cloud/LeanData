import Mathlib

namespace NUMINAMATH_CALUDE_complex_magnitude_l3076_307683

theorem complex_magnitude (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : a / (1 - i) = 1 - b * i) : 
  Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3076_307683


namespace NUMINAMATH_CALUDE_kiley_ate_two_slices_l3076_307648

/-- Represents a cheesecake with its properties and consumption -/
structure Cheesecake where
  calories_per_slice : ℕ
  total_calories : ℕ
  percent_eaten : ℚ

/-- Calculates the number of slices eaten given a Cheesecake -/
def slices_eaten (c : Cheesecake) : ℚ :=
  (c.total_calories / c.calories_per_slice : ℚ) * c.percent_eaten

/-- Theorem stating that Kiley ate 2 slices of the specified cheesecake -/
theorem kiley_ate_two_slices (c : Cheesecake) 
  (h1 : c.calories_per_slice = 350)
  (h2 : c.total_calories = 2800)
  (h3 : c.percent_eaten = 1/4) : 
  slices_eaten c = 2 := by
  sorry

end NUMINAMATH_CALUDE_kiley_ate_two_slices_l3076_307648


namespace NUMINAMATH_CALUDE_ab_nonpositive_l3076_307685

theorem ab_nonpositive (a b : ℚ) (ha : |a| = a) (hb : |b| = -b) : a * b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_nonpositive_l3076_307685


namespace NUMINAMATH_CALUDE_music_student_count_l3076_307675

/-- Represents the number of students in different categories -/
structure StudentCounts where
  total : ℕ
  art : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of students taking music -/
def musicStudents (counts : StudentCounts) : ℕ :=
  counts.total - counts.neither - (counts.art - counts.both)

/-- Theorem stating the number of students taking music -/
theorem music_student_count (counts : StudentCounts)
    (h_total : counts.total = 500)
    (h_art : counts.art = 10)
    (h_both : counts.both = 10)
    (h_neither : counts.neither = 470) :
    musicStudents counts = 30 := by
  sorry

#eval musicStudents { total := 500, art := 10, both := 10, neither := 470 }

end NUMINAMATH_CALUDE_music_student_count_l3076_307675


namespace NUMINAMATH_CALUDE_no_even_primes_greater_than_two_l3076_307611

theorem no_even_primes_greater_than_two :
  ∀ n : ℕ, n > 2 → Prime n → ¬Even n :=
by
  sorry

end NUMINAMATH_CALUDE_no_even_primes_greater_than_two_l3076_307611


namespace NUMINAMATH_CALUDE_rectangle_area_l3076_307690

theorem rectangle_area (length width diagonal : ℝ) (h1 : length / width = 5 / 2) (h2 : length^2 + width^2 = diagonal^2) (h3 : diagonal = 13) : 
  length * width = (10 / 29) * diagonal^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3076_307690


namespace NUMINAMATH_CALUDE_seventh_observation_seventh_observation_value_l3076_307658

theorem seventh_observation (initial_count : Nat) (initial_avg : ℝ) (new_avg : ℝ) : ℝ :=
  let total_count : Nat := initial_count + 1
  let initial_sum : ℝ := initial_count * initial_avg
  let new_sum : ℝ := total_count * new_avg
  new_sum - initial_sum

theorem seventh_observation_value :
  seventh_observation 6 12 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_seventh_observation_seventh_observation_value_l3076_307658


namespace NUMINAMATH_CALUDE_aquarium_species_count_l3076_307625

theorem aquarium_species_count 
  (sharks : ℕ) (eels : ℕ) (whales : ℕ) (dolphins : ℕ) (rays : ℕ) (octopuses : ℕ)
  (shark_pairs : ℕ) (eel_pairs : ℕ) (whale_pairs : ℕ) (octopus_split : ℕ)
  (h1 : sharks = 48)
  (h2 : eels = 21)
  (h3 : whales = 7)
  (h4 : dolphins = 16)
  (h5 : rays = 9)
  (h6 : octopuses = 30)
  (h7 : shark_pairs = 3)
  (h8 : eel_pairs = 2)
  (h9 : whale_pairs = 1)
  (h10 : octopus_split = 1) :
  sharks + eels + whales + dolphins + rays + octopuses 
  - (shark_pairs + eel_pairs + whale_pairs) 
  + octopus_split = 126 :=
by sorry

end NUMINAMATH_CALUDE_aquarium_species_count_l3076_307625


namespace NUMINAMATH_CALUDE_min_freight_cost_l3076_307647

/-- Represents the freight problem with given parameters -/
structure FreightProblem where
  totalOre : ℕ
  truckCapacity1 : ℕ
  truckCapacity2 : ℕ
  truckCost1 : ℕ
  truckCost2 : ℕ

/-- Calculates the total cost for a given number of trucks -/
def totalCost (p : FreightProblem) (trucks1 : ℕ) (trucks2 : ℕ) : ℕ :=
  trucks1 * p.truckCost1 + trucks2 * p.truckCost2

/-- Checks if a combination of trucks can transport the required amount of ore -/
def isValidCombination (p : FreightProblem) (trucks1 : ℕ) (trucks2 : ℕ) : Prop :=
  trucks1 * p.truckCapacity1 + trucks2 * p.truckCapacity2 ≥ p.totalOre

/-- The main theorem stating that 685 is the minimum freight cost -/
theorem min_freight_cost (p : FreightProblem) 
  (h1 : p.totalOre = 73)
  (h2 : p.truckCapacity1 = 7)
  (h3 : p.truckCapacity2 = 5)
  (h4 : p.truckCost1 = 65)
  (h5 : p.truckCost2 = 50) :
  (∀ trucks1 trucks2 : ℕ, isValidCombination p trucks1 trucks2 → totalCost p trucks1 trucks2 ≥ 685) ∧ 
  (∃ trucks1 trucks2 : ℕ, isValidCombination p trucks1 trucks2 ∧ totalCost p trucks1 trucks2 = 685) :=
sorry


end NUMINAMATH_CALUDE_min_freight_cost_l3076_307647


namespace NUMINAMATH_CALUDE_inequality_holds_l3076_307608

theorem inequality_holds (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : 0 < c) (h4 : c < 1) :
  b * (a ^ c) < a * (b ^ c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3076_307608


namespace NUMINAMATH_CALUDE_assembly_time_proof_l3076_307672

/-- Calculates the total time spent assembling furniture -/
def total_assembly_time (chairs tables time_per_piece : ℕ) : ℕ :=
  (chairs + tables) * time_per_piece

/-- Proves that given 20 chairs, 8 tables, and 6 minutes per piece, 
    the total assembly time is 168 minutes -/
theorem assembly_time_proof :
  total_assembly_time 20 8 6 = 168 := by
  sorry

end NUMINAMATH_CALUDE_assembly_time_proof_l3076_307672


namespace NUMINAMATH_CALUDE_function_inequality_l3076_307669

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x ∈ (Set.Ioo 0 (π/2)), f' x = deriv f x) →
  (∀ x ∈ (Set.Ioo 0 (π/2)), f x - f' x * (tan x) < 0) →
  (f 1 / sin 1 > 2 * f (π/6)) :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l3076_307669


namespace NUMINAMATH_CALUDE_inequality_proof_l3076_307638

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b ≥ 4) ∧ (a^2 + b^2 ≥ 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3076_307638


namespace NUMINAMATH_CALUDE_megan_pop_albums_l3076_307673

def country_albums : ℕ := 2
def songs_per_album : ℕ := 7
def total_songs : ℕ := 70

def pop_albums : ℕ := (total_songs - country_albums * songs_per_album) / songs_per_album

theorem megan_pop_albums : pop_albums = 8 := by
  sorry

end NUMINAMATH_CALUDE_megan_pop_albums_l3076_307673


namespace NUMINAMATH_CALUDE_parallel_lines_in_special_triangle_l3076_307693

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : Point)
  (b : Point)

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- Constructs an equilateral triangle given three points -/
def equilateral_triangle (a b c : Point) : Prop := sorry

theorem parallel_lines_in_special_triangle 
  (A B C M K : Point) 
  (h1 : equilateral_triangle A B C)
  (h2 : M.x ≥ A.x ∧ M.x ≤ B.x ∧ M.y = A.y)  -- M is on side AB
  (h3 : equilateral_triangle M K C) :
  parallel (Line.mk A C) (Line.mk B K) := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_in_special_triangle_l3076_307693


namespace NUMINAMATH_CALUDE_perpendicular_plane_line_not_always_perpendicular_l3076_307610

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_plane_line_not_always_perpendicular 
  (a b : Plane) (m : Line) :
  ¬(∀ (a b : Plane) (m : Line), 
    perpendicular a b ∧ contains a m → perp_line_plane m b) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_plane_line_not_always_perpendicular_l3076_307610


namespace NUMINAMATH_CALUDE_work_completion_time_a_completion_time_l3076_307602

/-- The time it takes for worker b to complete the work alone -/
def b_time : ℝ := 6

/-- The time it takes for worker b to complete the remaining work after both workers work for 1 day -/
def b_remaining_time : ℝ := 2.0000000000000004

/-- The time it takes for worker a to complete the work alone -/
def a_time : ℝ := 2

theorem work_completion_time :
  (1 / a_time + 1 / b_time) + b_remaining_time / b_time = 1 := by sorry

theorem a_completion_time : a_time = 2 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_a_completion_time_l3076_307602


namespace NUMINAMATH_CALUDE_income_tax_calculation_l3076_307614

def salary_jan_jun : ℕ := 23000
def salary_jul_dec : ℕ := 25000
def months_per_half_year : ℕ := 6
def prize_value : ℕ := 10000
def non_taxable_prize : ℕ := 4000
def salary_tax_rate : ℚ := 13 / 100
def prize_tax_rate : ℚ := 35 / 100

def total_income_tax : ℕ := 39540

theorem income_tax_calculation :
  let total_salary := salary_jan_jun * months_per_half_year + salary_jul_dec * months_per_half_year
  let salary_tax := (total_salary : ℚ) * salary_tax_rate
  let taxable_prize := prize_value - non_taxable_prize
  let prize_tax := (taxable_prize : ℚ) * prize_tax_rate
  let total_tax := salary_tax + prize_tax
  ⌊total_tax⌋ = total_income_tax := by sorry

end NUMINAMATH_CALUDE_income_tax_calculation_l3076_307614


namespace NUMINAMATH_CALUDE_y_coordinate_difference_zero_l3076_307699

/-- Given two points (m, n) and (m + 3, n + q) on the line x = (y / 7) - (2 / 5),
    the difference between their y-coordinates is 0. -/
theorem y_coordinate_difference_zero
  (m n q : ℚ) : 
  (m = n / 7 - 2 / 5) →
  (m + 3 = (n + q) / 7 - 2 / 5) →
  q = 0 :=
by sorry

end NUMINAMATH_CALUDE_y_coordinate_difference_zero_l3076_307699


namespace NUMINAMATH_CALUDE_geometry_propositions_l3076_307612

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem geometry_propositions 
  (a b : Line) (α β : Plane) :
  (∀ α β : Plane, ∀ a : Line, 
    parallel α β → subset a α → line_parallel a β) ∧
  (∀ a b : Line, ∀ α β : Plane,
    perpendicular a α → parallel α β → line_parallel b β → 
    line_perpendicular a b) := by
  sorry

end NUMINAMATH_CALUDE_geometry_propositions_l3076_307612


namespace NUMINAMATH_CALUDE_f_is_even_l3076_307607

def f (x : ℝ) : ℝ := 2 * x^2 - 1

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l3076_307607


namespace NUMINAMATH_CALUDE_grandma_olga_grandchildren_l3076_307662

/-- Represents the number of grandchildren Grandma Olga has. -/
def total_grandchildren : ℕ :=
  let daughters := 5
  let sons := 4
  let children_per_daughter := 8 + 7
  let children_per_son := 6 + 3
  daughters * children_per_daughter + sons * children_per_son

/-- Proves that Grandma Olga has 111 grandchildren. -/
theorem grandma_olga_grandchildren : total_grandchildren = 111 := by
  sorry

end NUMINAMATH_CALUDE_grandma_olga_grandchildren_l3076_307662


namespace NUMINAMATH_CALUDE_yard_area_l3076_307665

/-- The area of a rectangular yard with a rectangular cutout -/
theorem yard_area (length width cutout_length cutout_width : ℝ) 
  (h1 : length = 20)
  (h2 : width = 15)
  (h3 : cutout_length = 4)
  (h4 : cutout_width = 2) :
  length * width - cutout_length * cutout_width = 292 := by
  sorry

end NUMINAMATH_CALUDE_yard_area_l3076_307665


namespace NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l3076_307640

/-- A geometric sequence with a given product of its first five terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = a n * r

theorem third_term_of_geometric_sequence (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_product : a 1 * a 2 * a 3 * a 4 * a 5 = 32) : 
  a 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l3076_307640


namespace NUMINAMATH_CALUDE_concentric_circles_radius_l3076_307613

theorem concentric_circles_radius (r : ℝ) (R : ℝ) : 
  r > 0 → 
  (π * R^2) / (π * r^2) = 5 / 2 → 
  R = r * Real.sqrt 2.5 := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radius_l3076_307613


namespace NUMINAMATH_CALUDE_chef_wage_difference_chef_earns_less_l3076_307628

def manager_wage : ℚ := 17/2

theorem chef_wage_difference : ℚ :=
  let dishwasher_wage := manager_wage / 2
  let chef_wage := dishwasher_wage * (1 + 1/4)
  manager_wage - chef_wage

theorem chef_earns_less (h : chef_wage_difference = 255/80) : True := by
  sorry

end NUMINAMATH_CALUDE_chef_wage_difference_chef_earns_less_l3076_307628


namespace NUMINAMATH_CALUDE_opposite_of_sqrt3_plus_a_l3076_307649

theorem opposite_of_sqrt3_plus_a (a b : ℝ) (h : |a - 3*b| + Real.sqrt (b + 1) = 0) :
  -(Real.sqrt 3 + a) = 3 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt3_plus_a_l3076_307649


namespace NUMINAMATH_CALUDE_doll_count_l3076_307663

theorem doll_count (vera sophie aida : ℕ) : 
  vera = 20 → 
  sophie = 2 * vera → 
  aida = 2 * sophie → 
  vera + sophie + aida = 140 := by
sorry

end NUMINAMATH_CALUDE_doll_count_l3076_307663


namespace NUMINAMATH_CALUDE_sqrt_expressions_l3076_307660

theorem sqrt_expressions :
  (∀ x y z : ℝ, x = 27 ∧ y = 1/3 ∧ z = 3 → 
    Real.sqrt x - Real.sqrt y + Real.sqrt z = (11 * Real.sqrt 3) / 3) ∧
  (∀ a b c : ℝ, a = 32 ∧ b = 18 ∧ c = 2 → 
    (Real.sqrt a + Real.sqrt b) / Real.sqrt c - 8 = -1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_expressions_l3076_307660


namespace NUMINAMATH_CALUDE_intersection_equality_implies_complement_union_equality_l3076_307691

universe u

theorem intersection_equality_implies_complement_union_equality
  (U : Type u) [Nonempty U]
  (A B C : Set U)
  (h_nonempty_A : A.Nonempty)
  (h_nonempty_B : B.Nonempty)
  (h_nonempty_C : C.Nonempty)
  (h_intersection : A ∩ B = A ∩ C) :
  (Aᶜ ∪ B) = (Aᶜ ∪ C) :=
by sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_complement_union_equality_l3076_307691


namespace NUMINAMATH_CALUDE_correct_num_buckets_l3076_307666

/-- The number of crab buckets Tom has -/
def num_buckets : ℕ := 56

/-- The number of crabs in each bucket -/
def crabs_per_bucket : ℕ := 12

/-- The price of each crab in dollars -/
def price_per_crab : ℕ := 5

/-- Tom's weekly earnings in dollars -/
def weekly_earnings : ℕ := 3360

/-- Theorem stating that the number of crab buckets is correct -/
theorem correct_num_buckets : 
  num_buckets = weekly_earnings / (crabs_per_bucket * price_per_crab) := by
  sorry

end NUMINAMATH_CALUDE_correct_num_buckets_l3076_307666


namespace NUMINAMATH_CALUDE_coin_value_equality_l3076_307651

theorem coin_value_equality (n : ℕ) : 
  25 * 25 + 20 * 10 = 15 * 25 + 10 * 10 + n * 50 → n = 7 :=
by sorry

end NUMINAMATH_CALUDE_coin_value_equality_l3076_307651


namespace NUMINAMATH_CALUDE_man_walking_speed_percentage_l3076_307604

/-- Proves that a man is walking at 70% of his usual speed given the conditions -/
theorem man_walking_speed_percentage (usual_time distance : ℝ) 
  (h1 : usual_time = 56)
  (h2 : distance > 0)
  (h3 : distance = usual_time * (distance / usual_time)) -- Speed * Time = Distance
  (h4 : distance = 80 * (distance / (56 + 24))) -- New time is 80 minutes
  : (distance / (56 + 24)) / (distance / usual_time) = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_man_walking_speed_percentage_l3076_307604


namespace NUMINAMATH_CALUDE_original_number_proof_l3076_307676

/-- Given a three-digit number abc and N = 3194, where N is the sum of acb, bac, bca, cab, and cba, prove that abc = 358 -/
theorem original_number_proof (a b c : ℕ) (h1 : a ≠ 0) 
  (h2 : a * 100 + b * 10 + c < 1000) 
  (h3 : 3194 = (a * 100 + c * 10 + b) + (b * 100 + a * 10 + c) + 
               (b * 100 + c * 10 + a) + (c * 100 + a * 10 + b) + 
               (c * 100 + b * 10 + a)) : 
  a * 100 + b * 10 + c = 358 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l3076_307676


namespace NUMINAMATH_CALUDE_exists_unassemblable_configuration_l3076_307641

/-- Represents a rhombus divided into two triangles --/
structure Rhombus :=
  (white : Bool) -- true if the left triangle is white, false if it's gray

/-- Represents a rotation of the rhombus --/
inductive Rotation
  | R0   -- No rotation
  | R90  -- 90 degrees clockwise
  | R180 -- 180 degrees
  | R270 -- 270 degrees clockwise

/-- Represents a position in a 2D grid --/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a placed rhombus in a configuration --/
structure PlacedRhombus :=
  (rhombus : Rhombus)
  (rotation : Rotation)
  (position : Position)

/-- Represents a configuration of rhombuses --/
def Configuration := List PlacedRhombus

/-- Checks if a configuration is valid --/
def isValidConfiguration (config : Configuration) : Bool :=
  sorry

/-- Checks if a larger shape can be assembled from the given rhombuses --/
def canAssembleLargerShape (shape : Configuration) : Bool :=
  sorry

/-- The main theorem stating that there exists a configuration that cannot be assembled --/
theorem exists_unassemblable_configuration :
  ∃ (shape : Configuration),
    isValidConfiguration shape ∧ ¬canAssembleLargerShape shape :=
  sorry

end NUMINAMATH_CALUDE_exists_unassemblable_configuration_l3076_307641


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3076_307603

/-- The line y+2=k(x+1) always passes through the point (-1, -2) -/
theorem line_passes_through_point :
  ∀ (k : ℝ), ((-1) : ℝ) + 2 = k * ((-1) + 1) ∧ (-2 : ℝ) + 2 = k * ((-1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3076_307603


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3076_307656

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3076_307656


namespace NUMINAMATH_CALUDE_arrangement_count_correct_l3076_307615

def number_of_arrangements (men women : ℕ) : ℕ :=
  let first_group := men.choose 1 * women.choose 2
  let remaining_men := men - 1
  let remaining_women := women - 2
  let remaining_groups := remaining_men.choose 1 * remaining_women.choose 2
  first_group * remaining_groups

theorem arrangement_count_correct :
  number_of_arrangements 4 5 = 360 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_correct_l3076_307615


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3076_307657

theorem complex_equation_sum (a b : ℝ) :
  (a - 2 * Complex.I) * Complex.I = b + Complex.I →
  a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3076_307657


namespace NUMINAMATH_CALUDE_quadratic_equation_and_expression_calculation_l3076_307696

theorem quadratic_equation_and_expression_calculation :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 7 ∧ x₂ = 2 - Real.sqrt 7 ∧
    x₁^2 - 4*x₁ - 3 = 0 ∧ x₂^2 - 4*x₂ - 3 = 0) ∧
  (|-3| - 4 * Real.sin (π/4) + Real.sqrt 8 + (π - 3)^0 = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_and_expression_calculation_l3076_307696


namespace NUMINAMATH_CALUDE_x_equation_l3076_307667

theorem x_equation (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^11 - 7*x^7 + x^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_l3076_307667


namespace NUMINAMATH_CALUDE_apple_pricing_l3076_307621

/-- The cost per kilogram for the first 30 kgs of apples -/
def l : ℚ := 200 / 10

/-- The cost per kilogram for each additional kilogram after the first 30 kgs -/
def m : ℚ := 21

theorem apple_pricing :
  (l * 30 + m * 3 = 663) ∧
  (l * 30 + m * 6 = 726) ∧
  (l * 10 = 200) →
  m = 21 := by sorry

end NUMINAMATH_CALUDE_apple_pricing_l3076_307621


namespace NUMINAMATH_CALUDE_max_cab_value_l3076_307606

/-- Represents a two-digit number AB --/
def TwoDigitNumber (a b : Nat) : Prop :=
  10 ≤ 10 * a + b ∧ 10 * a + b < 100

/-- Represents a three-digit number CAB --/
def ThreeDigitNumber (c a b : Nat) : Prop :=
  100 ≤ 100 * c + 10 * a + b ∧ 100 * c + 10 * a + b < 1000

/-- The main theorem statement --/
theorem max_cab_value :
  ∀ a b c : Nat,
  a < 10 → b < 10 → c < 10 →
  TwoDigitNumber a b →
  ThreeDigitNumber c a b →
  (10 * a + b) * a = 100 * c + 10 * a + b →
  100 * c + 10 * a + b ≤ 895 :=
by sorry

end NUMINAMATH_CALUDE_max_cab_value_l3076_307606


namespace NUMINAMATH_CALUDE_unique_n_with_divisor_sum_property_l3076_307682

def isDivisor (d n : ℕ) : Prop := n % d = 0

theorem unique_n_with_divisor_sum_property :
  ∃! n : ℕ+, 
    (∃ (d₁ d₂ d₃ d₄ : ℕ+),
      isDivisor d₁ n ∧ isDivisor d₂ n ∧ isDivisor d₃ n ∧ isDivisor d₄ n ∧
      d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧
      d₁ = 1 ∧
      n = d₁^2 + d₂^2 + d₃^2 + d₄^2) ∧
    (∀ d : ℕ+, isDivisor d n → d = 1 ∨ d ≥ d₂) ∧
    n = 130 :=
by sorry

end NUMINAMATH_CALUDE_unique_n_with_divisor_sum_property_l3076_307682


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l3076_307600

theorem ratio_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l3076_307600


namespace NUMINAMATH_CALUDE_borrowed_sheets_average_l3076_307639

/-- Represents a document with pages printed on both sides of sheets. -/
structure Document where
  totalPages : Nat
  totalSheets : Nat
  pagesPerSheet : Nat
  borrowedSheets : Nat

/-- Calculates the average page number of remaining sheets after borrowing. -/
def averagePageNumber (doc : Document) : Rat :=
  let remainingSheets := doc.totalSheets - doc.borrowedSheets
  let totalPageSum := doc.totalPages * (doc.totalPages + 1) / 2
  let borrowedPagesStart := doc.borrowedSheets * doc.pagesPerSheet - (doc.pagesPerSheet - 1)
  let borrowedPagesEnd := doc.borrowedSheets * doc.pagesPerSheet
  let borrowedPageSum := (borrowedPagesStart + borrowedPagesEnd) * doc.borrowedSheets / 2
  (totalPageSum - borrowedPageSum) / remainingSheets

theorem borrowed_sheets_average (doc : Document) :
  doc.totalPages = 50 ∧
  doc.totalSheets = 25 ∧
  doc.pagesPerSheet = 2 ∧
  doc.borrowedSheets = 13 →
  averagePageNumber doc = 19 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sheets_average_l3076_307639


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3076_307618

theorem repeating_decimal_sum : 
  let x : ℚ := 2 / 9
  let y : ℚ := 1 / 33
  x + y = 25 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3076_307618


namespace NUMINAMATH_CALUDE_not_sum_solution_equation_example_sum_solution_equation_condition_l3076_307688

/-- Definition of a sum solution equation -/
def is_sum_solution_equation (a b : ℝ) : Prop :=
  (b / a) = b + a

/-- Theorem 1: 3x = 4.5 is not a sum solution equation -/
theorem not_sum_solution_equation_example : ¬ is_sum_solution_equation 3 4.5 := by
  sorry

/-- Theorem 2: 5x = m + 1 is a sum solution equation iff m = -29/4 -/
theorem sum_solution_equation_condition (m : ℝ) : 
  is_sum_solution_equation 5 (m + 1) ↔ m = -29/4 := by
  sorry

end NUMINAMATH_CALUDE_not_sum_solution_equation_example_sum_solution_equation_condition_l3076_307688


namespace NUMINAMATH_CALUDE_meaningful_expression_l3076_307680

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 1)) ↔ x > 1 := by
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3076_307680


namespace NUMINAMATH_CALUDE_gcd_18_30_l3076_307635

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l3076_307635


namespace NUMINAMATH_CALUDE_apple_cost_l3076_307678

/-- The cost of apples given specific pricing rules and total costs for certain weights. -/
theorem apple_cost (l q : ℝ) : 
  (∀ x, x ≤ 30 → x * l = x * 0.362) →  -- Cost for first 30 kgs
  (∀ x, x > 30 → x * l + (x - 30) * q = 30 * l + (x - 30) * q) →  -- Cost for additional kgs
  (33 * l + 3 * q = 11.67) →  -- Price for 33 kgs
  (36 * l + 6 * q = 12.48) →  -- Price for 36 kgs
  (10 * l = 3.62) :=  -- Cost of first 10 kgs
by sorry

end NUMINAMATH_CALUDE_apple_cost_l3076_307678


namespace NUMINAMATH_CALUDE_wickets_before_last_match_l3076_307646

/-- Represents the number of wickets taken before the last match -/
def W : ℕ := sorry

/-- The initial bowling average -/
def initial_average : ℚ := 12.4

/-- The number of wickets taken in the last match -/
def last_match_wickets : ℕ := 7

/-- The number of runs conceded in the last match -/
def last_match_runs : ℕ := 26

/-- The decrease in average after the last match -/
def average_decrease : ℚ := 0.4

/-- The new average after the last match -/
def new_average : ℚ := initial_average - average_decrease

theorem wickets_before_last_match :
  (initial_average * W + last_match_runs : ℚ) / (W + last_match_wickets) = new_average →
  W = 145 := by sorry

end NUMINAMATH_CALUDE_wickets_before_last_match_l3076_307646


namespace NUMINAMATH_CALUDE_birds_and_storks_on_fence_l3076_307670

theorem birds_and_storks_on_fence : 
  let initial_birds : ℕ := 2
  let additional_birds : ℕ := 5
  let storks : ℕ := 4
  let total_birds : ℕ := initial_birds + additional_birds
  (total_birds - storks) = 3 := by
  sorry

end NUMINAMATH_CALUDE_birds_and_storks_on_fence_l3076_307670


namespace NUMINAMATH_CALUDE_train_crossing_time_l3076_307645

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 135 →
  train_speed_kmh = 54 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 9 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3076_307645


namespace NUMINAMATH_CALUDE_triangles_equality_l3076_307624

-- Define the points
variable (A K L M N G G' : ℝ × ℝ)

-- Define the angle α
variable (α : ℝ)

-- Define similarity of triangles
def similar_triangles (t1 t2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

-- Define isosceles triangle
def isosceles_triangle (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

-- Define angle at vertex
def angle_at_vertex (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) (v : ℝ × ℝ) (θ : ℝ) : Prop := sorry

-- State the theorem
theorem triangles_equality (h1 : similar_triangles (A, K, L) (A, M, N))
                           (h2 : isosceles_triangle (A, K, L))
                           (h3 : isosceles_triangle (A, M, N))
                           (h4 : angle_at_vertex (A, K, L) A α)
                           (h5 : angle_at_vertex (A, M, N) A α)
                           (h6 : similar_triangles (G, N, K) (G', L, M))
                           (h7 : isosceles_triangle (G, N, K))
                           (h8 : isosceles_triangle (G', L, M))
                           (h9 : angle_at_vertex (G, N, K) G (π - α))
                           (h10 : angle_at_vertex (G', L, M) G' (π - α)) :
  G = G' := by sorry

end NUMINAMATH_CALUDE_triangles_equality_l3076_307624


namespace NUMINAMATH_CALUDE_grid_path_count_l3076_307668

/-- The number of paths on a grid from (0,0) to (m,n) using only right and up moves -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The dimensions of our grid -/
def gridWidth : ℕ := 6
def gridHeight : ℕ := 5

/-- The total number of steps required -/
def totalSteps : ℕ := gridWidth + gridHeight

theorem grid_path_count :
  gridPaths gridWidth gridHeight = 462 := by
  sorry

#eval gridPaths gridWidth gridHeight

end NUMINAMATH_CALUDE_grid_path_count_l3076_307668


namespace NUMINAMATH_CALUDE_standard_poodle_height_difference_l3076_307689

/-- The height difference between the standard poodle and the miniature poodle -/
def height_difference (standard_height miniature_height : ℕ) : ℕ :=
  standard_height - miniature_height

/-- Theorem: The standard poodle is 8 inches taller than the miniature poodle -/
theorem standard_poodle_height_difference :
  let toy_height : ℕ := 14
  let standard_height : ℕ := 28
  let miniature_height : ℕ := toy_height + 6
  height_difference standard_height miniature_height = 8 := by
  sorry

end NUMINAMATH_CALUDE_standard_poodle_height_difference_l3076_307689


namespace NUMINAMATH_CALUDE_rate_of_discount_l3076_307629

/-- Calculate the rate of discount given the marked price and selling price -/
theorem rate_of_discount (marked_price selling_price : ℝ) 
  (h1 : marked_price = 150)
  (h2 : selling_price = 120) : 
  (marked_price - selling_price) / marked_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rate_of_discount_l3076_307629


namespace NUMINAMATH_CALUDE_glasses_wearers_properties_l3076_307637

-- Define the universe of women
variable (Woman : Type)

-- Define predicates
variable (wears_glasses : Woman → Prop)
variable (knows_english : Woman → Prop)
variable (wears_chignon : Woman → Prop)
variable (has_seven_children : Woman → Prop)

-- Define the five statements as axioms
axiom statement1 : ∀ w : Woman, has_seven_children w → knows_english w → wears_chignon w
axiom statement2 : ∀ w : Woman, wears_glasses w → (has_seven_children w ∨ knows_english w)
axiom statement3 : ∀ w : Woman, ¬has_seven_children w → wears_glasses w → wears_chignon w
axiom statement4 : ∀ w : Woman, has_seven_children w → wears_glasses w → knows_english w
axiom statement5 : ∀ w : Woman, wears_chignon w → ¬has_seven_children w

-- Theorem to prove
theorem glasses_wearers_properties :
  ∀ w : Woman, wears_glasses w → (knows_english w ∧ wears_chignon w ∧ ¬has_seven_children w) := by
  sorry

end NUMINAMATH_CALUDE_glasses_wearers_properties_l3076_307637


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3076_307630

theorem sufficient_not_necessary (a b : ℝ) : 
  (a < 0 ∧ -1 < b ∧ b < 0) → 
  (∀ x y : ℝ, x < 0 ∧ -1 < y ∧ y < 0 → x + x * y < 0) ∧
  ¬(∀ x y : ℝ, x + x * y < 0 → x < 0 ∧ -1 < y ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3076_307630


namespace NUMINAMATH_CALUDE_max_missed_problems_l3076_307692

theorem max_missed_problems (total_problems : ℕ) (passing_percentage : ℚ) : 
  total_problems = 50 → 
  passing_percentage = 75/100 → 
  ∃ (max_missed : ℕ), max_missed = 12 ∧ 
    (∀ (missed : ℕ), missed ≤ max_missed → 
      (total_problems - missed) / total_problems ≥ passing_percentage) ∧
    (∀ (missed : ℕ), missed > max_missed → 
      (total_problems - missed) / total_problems < passing_percentage) :=
by sorry

end NUMINAMATH_CALUDE_max_missed_problems_l3076_307692


namespace NUMINAMATH_CALUDE_infinitely_many_invalid_d_l3076_307687

/-- The perimeter difference between the triangle and rectangle -/
def perimeter_difference : ℕ := 504

/-- The length of the shorter side of the rectangle -/
def rectangle_short_side : ℕ := 7

/-- Represents the relationship between the triangle side length, rectangle long side, and d -/
def triangle_rectangle_relation (triangle_side : ℝ) (rectangle_long_side : ℝ) (d : ℝ) : Prop :=
  triangle_side = rectangle_long_side + d

/-- Represents the perimeter relationship between the triangle and rectangle -/
def perimeter_relation (triangle_side : ℝ) (rectangle_long_side : ℝ) : Prop :=
  3 * triangle_side - 2 * (rectangle_long_side + rectangle_short_side) = perimeter_difference

/-- The main theorem stating that there are infinitely many positive integers
    that cannot be valid values for d -/
theorem infinitely_many_invalid_d : ∃ (S : Set ℕ), Set.Infinite S ∧
  ∀ (d : ℕ), d ∈ S →
    ¬∃ (triangle_side rectangle_long_side : ℝ),
      triangle_rectangle_relation triangle_side rectangle_long_side d ∧
      perimeter_relation triangle_side rectangle_long_side :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_invalid_d_l3076_307687


namespace NUMINAMATH_CALUDE_two_hundred_fiftieth_term_is_331_l3076_307654

/-- The sequence function that generates the nth term of the sequence 
    by omitting perfect squares and multiples of 5 -/
def sequenceFunction (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the 250th term of the sequence is 331 -/
theorem two_hundred_fiftieth_term_is_331 : sequenceFunction 250 = 331 := by
  sorry

end NUMINAMATH_CALUDE_two_hundred_fiftieth_term_is_331_l3076_307654


namespace NUMINAMATH_CALUDE_triangle_properties_l3076_307697

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def satisfies_conditions (t : Triangle) : Prop :=
  t.c = 2 * Real.sqrt 3 ∧
  t.a * Real.sin t.A - t.c * Real.sin t.C = (t.a - t.b) * Real.sin t.B ∧
  t.c + t.b * Real.cos t.A = t.a * (4 * Real.cos t.A + Real.cos t.B)

/-- Theorem stating the conclusions -/
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.C = Real.pi / 3 ∧ t.a * t.b * Real.sin t.C / 2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3076_307697


namespace NUMINAMATH_CALUDE_sum_of_squares_l3076_307627

theorem sum_of_squares (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 13) : a^2 + b^2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3076_307627


namespace NUMINAMATH_CALUDE_prob_two_girls_l3076_307620

/-- The probability of selecting two girls from a group of 15 members, where 6 are girls -/
theorem prob_two_girls (total : ℕ) (girls : ℕ) (h1 : total = 15) (h2 : girls = 6) :
  (girls.choose 2 : ℚ) / (total.choose 2) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_girls_l3076_307620


namespace NUMINAMATH_CALUDE_decreasing_quadratic_range_l3076_307661

/-- A quadratic function f(x) with parameter a. -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The theorem stating that if f(x) is decreasing on (-∞, 4], then a ≤ -3. -/
theorem decreasing_quadratic_range (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) →
  a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_range_l3076_307661


namespace NUMINAMATH_CALUDE_percent_equality_l3076_307695

theorem percent_equality (x : ℝ) (h : 0.30 * 0.15 * x = 45) : 0.15 * 0.30 * x = 45 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l3076_307695


namespace NUMINAMATH_CALUDE_length_AG_l3076_307694

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right-angled at A
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- AB = 3
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 9 ∧
  -- AC = 3√3
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 27

-- Define the altitude AD
def Altitude (A B C D : ℝ × ℝ) : Prop :=
  (D.1 - A.1) * (B.1 - C.1) + (D.2 - A.2) * (B.2 - C.2) = 0

-- Define the median AM
def Median (A B C M : ℝ × ℝ) : Prop :=
  M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the intersection point G
def Intersection (A D M G : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, G = (A.1 + t * (D.1 - A.1), A.2 + t * (D.2 - A.2)) ∧
             ∃ s : ℝ, G = (A.1 + s * (M.1 - A.1), A.2 + s * (M.2 - A.2))

-- Theorem statement
theorem length_AG (A B C D M G : ℝ × ℝ) :
  Triangle A B C →
  Altitude A B C D →
  Median A B C M →
  Intersection A D M G →
  (G.1 - A.1)^2 + (G.2 - A.2)^2 = 243/64 :=
by sorry

end NUMINAMATH_CALUDE_length_AG_l3076_307694


namespace NUMINAMATH_CALUDE_stone_122_is_9_l3076_307636

/-- Represents the number of stones in the line -/
def n : ℕ := 17

/-- The target count we're looking for -/
def target : ℕ := 122

/-- Function to determine the original stone number given a count in the sequence -/
def originalStone (count : ℕ) : ℕ :=
  let modulo := count % (2 * (n - 1))
  if modulo ≤ n then
    modulo
  else
    2 * n - modulo

/-- Theorem stating that the stone counted as 122 is originally stone number 9 -/
theorem stone_122_is_9 : originalStone target = 9 := by
  sorry


end NUMINAMATH_CALUDE_stone_122_is_9_l3076_307636


namespace NUMINAMATH_CALUDE_no_geometric_sequence_satisfies_conditions_l3076_307632

theorem no_geometric_sequence_satisfies_conditions :
  ¬ ∃ (a : ℕ → ℝ) (q : ℝ),
    (∀ n : ℕ, a (n + 1) = q * a n) ∧  -- geometric sequence
    (a 1 + a 6 = 11) ∧  -- condition 1
    (a 3 * a 4 = 32 / 9) ∧  -- condition 1
    (∀ n : ℕ, a (n + 1) > a n) ∧  -- condition 2
    (∃ m : ℕ, m > 4 ∧ 
      2 * (a m)^2 = 2/3 * a (m - 1) + (a (m + 1) + 4/9)) :=  -- condition 3
by sorry

end NUMINAMATH_CALUDE_no_geometric_sequence_satisfies_conditions_l3076_307632


namespace NUMINAMATH_CALUDE_donny_gas_station_payment_l3076_307644

/-- Given the conditions of Donny's gas station visit, prove that he paid $350. -/
theorem donny_gas_station_payment (tank_capacity : ℝ) (initial_fuel : ℝ) (fuel_cost : ℝ) (change : ℝ)
  (h1 : tank_capacity = 150)
  (h2 : initial_fuel = 38)
  (h3 : fuel_cost = 3)
  (h4 : change = 14) :
  (tank_capacity - initial_fuel) * fuel_cost + change = 350 := by
  sorry

#check donny_gas_station_payment

end NUMINAMATH_CALUDE_donny_gas_station_payment_l3076_307644


namespace NUMINAMATH_CALUDE_tetrahedron_colorings_l3076_307698

/-- Represents a coloring of the tetrahedron -/
def Coloring := Fin 7 → Bool

/-- The group of rotational symmetries of a tetrahedron -/
def TetrahedronSymmetry : Type := Unit -- Placeholder, actual implementation would be more complex

/-- Action of a symmetry on a coloring -/
def symmetryAction (s : TetrahedronSymmetry) (c : Coloring) : Coloring :=
  sorry

/-- A coloring is considered fixed under a symmetry if it's unchanged by the symmetry's action -/
def isFixed (s : TetrahedronSymmetry) (c : Coloring) : Prop :=
  symmetryAction s c = c

/-- The number of distinct colorings under rotational symmetry -/
def numDistinctColorings : ℕ :=
  sorry

theorem tetrahedron_colorings : numDistinctColorings = 48 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_colorings_l3076_307698


namespace NUMINAMATH_CALUDE_selling_price_equal_profit_loss_l3076_307655

def cost_price : ℕ := 49
def loss_price : ℕ := 42

def profit (selling_price : ℕ) : ℤ := selling_price - cost_price
def loss (selling_price : ℕ) : ℤ := cost_price - selling_price

theorem selling_price_equal_profit_loss : 
  ∃ (sp : ℕ), profit sp = loss loss_price ∧ profit sp > 0 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_equal_profit_loss_l3076_307655


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_l3076_307643

/-- An arithmetic sequence with its first n terms sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_m (seq : ArithmeticSequence) (m : ℕ) :
  m ≥ 2 →
  seq.S (m - 1) = -2 →
  seq.S m = 0 →
  seq.S (m + 1) = 3 →
  m = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_m_l3076_307643


namespace NUMINAMATH_CALUDE_three_digit_sum_permutations_l3076_307609

/-- Given a three-digit number m = 100a + 10b + c, where a, b, and c are single digits,
    if the sum of m and its five permutations (acb), (bca), (bac), (cab), and (cba) is 3315,
    then m = 015. -/
theorem three_digit_sum_permutations (a b c : ℕ) : 
  (0 ≤ a ∧ a ≤ 9) → 
  (0 ≤ b ∧ b ≤ 9) → 
  (0 ≤ c ∧ c ≤ 9) → 
  let m := 100 * a + 10 * b + c
  (m + (100 * a + 10 * c + b) + (100 * b + 10 * c + a) + 
   (100 * b + 10 * a + c) + (100 * c + 10 * a + b) + 
   (100 * c + 10 * b + a) = 3315) →
  m = 15 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_permutations_l3076_307609


namespace NUMINAMATH_CALUDE_line_equation_correct_l3076_307659

/-- Represents a 2D point -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

def Point2D.toVector (p : Point2D) : Vector2D :=
  { x := p.x, y := p.y }

def isPointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def hasDirection (l : Line2D) (v : Vector2D) : Prop :=
  l.a * v.y = -l.b * v.x

theorem line_equation_correct (A : Point2D) (n : Vector2D) (l : Line2D) :
  A.x = 3 ∧ A.y = -2 ∧ n.x = -5 ∧ n.y = 3 ∧ 
  l.a = 3 ∧ l.b = 5 ∧ l.c = 1 →
  isPointOnLine A l ∧ hasDirection l n := by
  sorry

end NUMINAMATH_CALUDE_line_equation_correct_l3076_307659


namespace NUMINAMATH_CALUDE_contractor_male_workers_l3076_307677

/-- Proves that the number of male workers is 20 given the conditions of the problem -/
theorem contractor_male_workers :
  let female_workers : ℕ := 15
  let child_workers : ℕ := 5
  let male_wage : ℚ := 25
  let female_wage : ℚ := 20
  let child_wage : ℚ := 8
  let average_wage : ℚ := 21
  ∃ male_workers : ℕ,
    (male_wage * male_workers + female_wage * female_workers + child_wage * child_workers) /
    (male_workers + female_workers + child_workers) = average_wage ∧
    male_workers = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_contractor_male_workers_l3076_307677


namespace NUMINAMATH_CALUDE_sum_of_roots_sum_of_roots_is_twenty_l3076_307642

/-- Square with sides parallel to coordinate axes -/
structure Square :=
  (side_length : ℝ)
  (bottom_left : ℝ × ℝ)

/-- Parabola defined by y = (1/5)x^2 + ax + b -/
structure Parabola :=
  (a : ℝ)
  (b : ℝ)

/-- Configuration of square and parabola -/
structure Configuration :=
  (square : Square)
  (parabola : Parabola)
  (passes_through_B : Bool)
  (passes_through_C : Bool)
  (vertex_on_AD : Bool)

/-- Theorem: Sum of roots of quadratic polynomial -/
theorem sum_of_roots (config : Configuration) : ℝ :=
  20

/-- Main theorem: Sum of roots is 20 -/
theorem sum_of_roots_is_twenty (config : Configuration) :
  sum_of_roots config = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_sum_of_roots_is_twenty_l3076_307642


namespace NUMINAMATH_CALUDE_zero_subset_X_l3076_307605

def X : Set ℝ := {x | x > -1}

theorem zero_subset_X : {0} ⊆ X := by
  sorry

end NUMINAMATH_CALUDE_zero_subset_X_l3076_307605


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3076_307674

def arithmetic_sequence : List ℕ := [71, 75, 79, 83, 87, 91]

theorem arithmetic_sequence_sum : 
  3 * (arithmetic_sequence.sum) = 1458 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3076_307674


namespace NUMINAMATH_CALUDE_modulus_of_z_l3076_307664

/-- The modulus of the complex number z = 2/(1-i) + (1-i)^2 is equal to √2 -/
theorem modulus_of_z (i : ℂ) (h : i^2 = -1) :
  Complex.abs (2 / (1 - i) + (1 - i)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3076_307664


namespace NUMINAMATH_CALUDE_yoongi_has_fewest_apples_l3076_307684

def yoongi_apples : ℕ := 4
def yuna_apples : ℕ := 5
def jungkook_apples : ℕ := 6 * 3

theorem yoongi_has_fewest_apples :
  yoongi_apples < yuna_apples ∧ yoongi_apples < jungkook_apples :=
by sorry

end NUMINAMATH_CALUDE_yoongi_has_fewest_apples_l3076_307684


namespace NUMINAMATH_CALUDE_clock_problem_l3076_307623

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Represents duration in hours, minutes, and seconds -/
structure Duration where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Converts total seconds to Time structure -/
def secondsToTime (totalSeconds : Nat) : Time :=
  let hours := totalSeconds / 3600
  let minutes := (totalSeconds % 3600) / 60
  let seconds := totalSeconds % 60
  { hours := hours % 12, minutes := minutes, seconds := seconds }

/-- Adds a Duration to a Time, wrapping around 12-hour clock -/
def addDuration (t : Time) (d : Duration) : Time :=
  let totalSeconds := 
    (t.hours * 3600 + t.minutes * 60 + t.seconds) +
    (d.hours * 3600 + d.minutes * 60 + d.seconds)
  secondsToTime totalSeconds

/-- Calculates the sum of digits in a Time -/
def sumDigits (t : Time) : Nat :=
  t.hours + t.minutes + t.seconds

theorem clock_problem (initialTime : Time) (elapsedTime : Duration) : 
  initialTime.hours = 3 ∧ 
  initialTime.minutes = 15 ∧ 
  initialTime.seconds = 20 ∧
  elapsedTime.hours = 305 ∧ 
  elapsedTime.minutes = 45 ∧ 
  elapsedTime.seconds = 56 →
  let finalTime := addDuration initialTime elapsedTime
  finalTime.hours = 9 ∧ 
  finalTime.minutes = 1 ∧ 
  finalTime.seconds = 16 ∧
  sumDigits finalTime = 26 := by
  sorry

end NUMINAMATH_CALUDE_clock_problem_l3076_307623


namespace NUMINAMATH_CALUDE_solution_to_system_l3076_307619

theorem solution_to_system (x y : ℝ) : 
  (x^2*y - x*y^2 - 5*x + 5*y + 3 = 0 ∧ 
   x^3*y - x*y^3 - 5*x^2 + 5*y^2 + 15 = 0) ↔ 
  (x = 4 ∧ y = 1) := by sorry

end NUMINAMATH_CALUDE_solution_to_system_l3076_307619


namespace NUMINAMATH_CALUDE_daily_earnings_of_c_l3076_307634

theorem daily_earnings_of_c (A B C : ℕ) 
  (h1 : A + B + C = 600)
  (h2 : A + C = 400)
  (h3 : B + C = 300) :
  C = 100 := by
sorry

end NUMINAMATH_CALUDE_daily_earnings_of_c_l3076_307634


namespace NUMINAMATH_CALUDE_last_gift_probability_theorem_l3076_307633

/-- Represents a circular arrangement of houses -/
structure CircularArrangement where
  numHouses : ℕ
  startHouse : ℕ

/-- Probability of moving to either neighbor -/
def moveProbability : ℚ := 1/2

/-- The probability that a specific house is the last to receive a gift -/
def lastGiftProbability (ca : CircularArrangement) : ℚ :=
  1 / (ca.numHouses - 1 : ℚ)

theorem last_gift_probability_theorem (ca : CircularArrangement) 
  (h1 : ca.numHouses = 2014) 
  (h2 : ca.startHouse < ca.numHouses) 
  (h3 : moveProbability = 1/2) :
  lastGiftProbability ca = 1/2013 := by
  sorry

end NUMINAMATH_CALUDE_last_gift_probability_theorem_l3076_307633


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l3076_307681

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- The vertices of the triangle -/
def A : ℝ × ℝ × ℝ := (2, 3, 4)
def B : ℝ × ℝ × ℝ := (6, 4, 2)
def C : ℝ × ℝ × ℝ := (4, 6, 6)

/-- Theorem: The orthocenter of triangle ABC is (10/7, 51/7, 12/7) -/
theorem orthocenter_of_triangle :
  orthocenter A B C = (10/7, 51/7, 12/7) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l3076_307681


namespace NUMINAMATH_CALUDE_symmetric_point_xoz_l3076_307671

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOz plane -/
def xOzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Symmetry with respect to the xOz plane -/
def symmetricPointXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetric_point_xoz (p : Point3D) :
  p.x = 2 ∧ p.y = 1 ∧ p.z = 3 →
  symmetricPointXOZ p = Point3D.mk 2 (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_xoz_l3076_307671


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3076_307652

theorem quadratic_equation_solution :
  let a : ℝ := 1
  let b : ℝ := 5
  let c : ℝ := -1
  let x₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ = (-5 + Real.sqrt 29) / 2 ∧
  x₂ = (-5 - Real.sqrt 29) / 2 ∧
  a * x₁^2 + b * x₁ + c = 0 ∧
  a * x₂^2 + b * x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3076_307652


namespace NUMINAMATH_CALUDE_inequality_solution_l3076_307653

theorem inequality_solution (x : ℝ) : 
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + x^2)) ↔ x ≥ -12 / 7 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3076_307653


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3076_307650

theorem trigonometric_identities :
  -- Part 1
  (Real.sin (76 * π / 180) * Real.cos (74 * π / 180) + Real.sin (14 * π / 180) * Real.cos (16 * π / 180) = 1/2) ∧
  -- Part 2
  ((1 - Real.tan (59 * π / 180)) * (1 - Real.tan (76 * π / 180)) = 2) ∧
  -- Part 3
  ((Real.sin (7 * π / 180) + Real.cos (15 * π / 180) * Real.sin (8 * π / 180)) / 
   (Real.cos (7 * π / 180) - Real.sin (15 * π / 180) * Real.sin (8 * π / 180)) = 2 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3076_307650


namespace NUMINAMATH_CALUDE_problem_part1_l3076_307622

theorem problem_part1 : (-2)^2 + |Real.sqrt 2 - 1| - Real.sqrt 4 = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_part1_l3076_307622


namespace NUMINAMATH_CALUDE_sharmila_hourly_wage_l3076_307626

/-- Represents Sharmila's work schedule and earnings -/
structure WorkSchedule where
  monday_hours : ℕ
  wednesday_hours : ℕ
  friday_hours : ℕ
  tuesday_hours : ℕ
  thursday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week -/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.monday_hours + 2 * schedule.tuesday_hours

/-- Calculates the hourly wage given a work schedule -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sharmila's work schedule -/
def sharmila_schedule : WorkSchedule := {
  monday_hours := 10,
  wednesday_hours := 10,
  friday_hours := 10,
  tuesday_hours := 8,
  thursday_hours := 8,
  weekly_earnings := 460
}

/-- Theorem stating that Sharmila's hourly wage is $10 -/
theorem sharmila_hourly_wage :
  hourly_wage sharmila_schedule = 10 := by sorry

end NUMINAMATH_CALUDE_sharmila_hourly_wage_l3076_307626


namespace NUMINAMATH_CALUDE_minimum_value_reciprocal_sum_l3076_307616

theorem minimum_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geo_mean : Real.sqrt 5 = Real.sqrt (5^a * 5^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_reciprocal_sum_l3076_307616


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_678_l3076_307686

theorem sin_n_equals_cos_678 (n : ℤ) (h1 : -120 ≤ n) (h2 : n ≤ 120) :
  Real.sin (n * π / 180) = Real.cos (678 * π / 180) → n = 48 := by
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_678_l3076_307686


namespace NUMINAMATH_CALUDE_pet_food_sale_discount_l3076_307617

def msrp : ℝ := 45.00
def max_regular_discount : ℝ := 0.30
def min_sale_price : ℝ := 25.20

theorem pet_food_sale_discount : ∃ (additional_discount : ℝ),
  additional_discount = 0.20 ∧
  min_sale_price = msrp * (1 - max_regular_discount) * (1 - additional_discount) :=
sorry

end NUMINAMATH_CALUDE_pet_food_sale_discount_l3076_307617


namespace NUMINAMATH_CALUDE_impossible_to_reach_all_plus_l3076_307679

/- Define the sign type -/
inductive Sign : Type
| Plus : Sign
| Minus : Sign

/- Define the 4x4 grid type -/
def Grid := Matrix (Fin 4) (Fin 4) Sign

/- Define the initial grid -/
def initial_grid : Grid :=
  λ i j => match i, j with
  | 0, 1 => Sign.Minus
  | 3, 1 => Sign.Minus
  | _, _ => Sign.Plus

/- Define a move (flipping a row or column) -/
def flip_row (g : Grid) (row : Fin 4) : Grid :=
  λ i j => if i = row then
    match g i j with
    | Sign.Plus => Sign.Minus
    | Sign.Minus => Sign.Plus
    else g i j

def flip_column (g : Grid) (col : Fin 4) : Grid :=
  λ i j => if j = col then
    match g i j with
    | Sign.Plus => Sign.Minus
    | Sign.Minus => Sign.Plus
    else g i j

/- Define the goal state (all Plus signs) -/
def all_plus (g : Grid) : Prop :=
  ∀ i j, g i j = Sign.Plus

/- The main theorem -/
theorem impossible_to_reach_all_plus :
  ¬ ∃ (moves : List (Sum (Fin 4) (Fin 4))),
    all_plus (moves.foldl (λ g move => match move with
      | Sum.inl row => flip_row g row
      | Sum.inr col => flip_column g col) initial_grid) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_reach_all_plus_l3076_307679


namespace NUMINAMATH_CALUDE_cos_sum_squared_one_solutions_l3076_307601

theorem cos_sum_squared_one_solutions (x : ℝ) : 
  (Real.cos x)^2 + (Real.cos (2*x))^2 + (Real.cos (3*x))^2 = 1 ↔ 
  (∃ k : ℤ, x = π/2 + k*π ∨ 
            x = π/4 + 2*k*π ∨ 
            x = 3*π/4 + 2*k*π ∨ 
            x = π/6 + 2*k*π ∨ 
            x = 5*π/6 + 2*k*π) :=
by sorry

end NUMINAMATH_CALUDE_cos_sum_squared_one_solutions_l3076_307601


namespace NUMINAMATH_CALUDE_chloe_final_score_is_86_l3076_307631

/-- Chloe's final score in a trivia game -/
def chloeFinalScore (firstRoundScore secondRoundScore lastRoundLoss : ℕ) : ℕ :=
  firstRoundScore + secondRoundScore - lastRoundLoss

/-- Theorem: Chloe's final score is 86 points -/
theorem chloe_final_score_is_86 :
  chloeFinalScore 40 50 4 = 86 := by
  sorry

#eval chloeFinalScore 40 50 4

end NUMINAMATH_CALUDE_chloe_final_score_is_86_l3076_307631
