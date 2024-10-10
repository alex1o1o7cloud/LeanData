import Mathlib

namespace cheerleader_size6_count_l458_45869

/-- Represents the number of cheerleaders needing each uniform size -/
structure CheerleaderSizes where
  size2 : ℕ
  size6 : ℕ
  size12 : ℕ

/-- The conditions of the cheerleader uniform problem -/
def cheerleader_uniform_problem (s : CheerleaderSizes) : Prop :=
  s.size2 = 4 ∧
  s.size12 * 2 = s.size6 ∧
  s.size2 + s.size6 + s.size12 = 19

/-- The theorem stating the solution to the cheerleader uniform problem -/
theorem cheerleader_size6_count :
  ∃ s : CheerleaderSizes, cheerleader_uniform_problem s ∧ s.size6 = 10 :=
sorry

end cheerleader_size6_count_l458_45869


namespace average_donation_l458_45805

theorem average_donation (total_people : ℝ) (h_total_positive : total_people > 0) : 
  let group1_fraction : ℝ := 1 / 10
  let group2_fraction : ℝ := 3 / 4
  let group3_fraction : ℝ := 1 - group1_fraction - group2_fraction
  let donation1 : ℝ := 200
  let donation2 : ℝ := 100
  let donation3 : ℝ := 50
  let total_donation : ℝ := 
    group1_fraction * donation1 * total_people + 
    group2_fraction * donation2 * total_people + 
    group3_fraction * donation3 * total_people
  total_donation / total_people = 102.5 := by
sorry

end average_donation_l458_45805


namespace mona_unique_players_l458_45802

/-- The number of unique players Mona grouped with in a video game --/
def unique_players (groups : ℕ) (players_per_group : ℕ) (repeated_players : ℕ) : ℕ :=
  groups * players_per_group - repeated_players

/-- Theorem stating the number of unique players Mona grouped with --/
theorem mona_unique_players :
  let groups : ℕ := 9
  let players_per_group : ℕ := 4
  let repeated_players : ℕ := 3
  unique_players groups players_per_group repeated_players = 33 := by
  sorry

#eval unique_players 9 4 3

end mona_unique_players_l458_45802


namespace expand_and_simplify_l458_45852

theorem expand_and_simplify (x : ℝ) :
  5 * (6 * x^3 - 3 * x^2 + 4 * x - 2) = 30 * x^3 - 15 * x^2 + 20 * x - 10 := by
  sorry

end expand_and_simplify_l458_45852


namespace mike_seed_count_l458_45828

theorem mike_seed_count (seeds_left : ℕ) (seeds_to_left : ℕ) (seeds_to_new : ℕ) 
  (h1 : seeds_to_left = 20)
  (h2 : seeds_left = 30)
  (h3 : seeds_to_new = 30) :
  seeds_left + seeds_to_left + 2 * seeds_to_left + seeds_to_new = 120 := by
  sorry

#check mike_seed_count

end mike_seed_count_l458_45828


namespace christmas_decorations_distribution_l458_45855

theorem christmas_decorations_distribution :
  let total_decorations : ℕ := 455
  let valid_student_count (n : ℕ) : Prop := 10 < n ∧ n < 70
  let valid_distribution (students : ℕ) (per_student : ℕ) : Prop :=
    valid_student_count students ∧
    students * per_student = total_decorations

  (∀ students per_student, valid_distribution students per_student →
    (students = 65 ∧ per_student = 7) ∨
    (students = 35 ∧ per_student = 13) ∨
    (students = 13 ∧ per_student = 35)) ∧
  (valid_distribution 65 7 ∧
   valid_distribution 35 13 ∧
   valid_distribution 13 35) :=
by sorry

end christmas_decorations_distribution_l458_45855


namespace max_contribution_l458_45842

theorem max_contribution 
  (total : ℝ) 
  (num_people : ℕ) 
  (min_contribution : ℝ) 
  (h1 : total = 20) 
  (h2 : num_people = 12) 
  (h3 : min_contribution = 1) 
  (h4 : ∀ p, p ≤ num_people → p • min_contribution ≤ total) : 
  ∃ max_contrib : ℝ, max_contrib = 9 ∧ 
    ∀ individual_contrib, 
      individual_contrib ≤ max_contrib ∧ 
      (num_people - 1) • min_contribution + individual_contrib = total :=
sorry

end max_contribution_l458_45842


namespace circle_segment_area_l458_45877

theorem circle_segment_area (R : ℝ) (R_pos : R > 0) : 
  let circle_area := π * R^2
  let square_side := R * Real.sqrt 2
  let square_area := square_side^2
  let segment_area := (circle_area - square_area) / 4
  segment_area = R^2 * (π - 2) / 4 := by
sorry

end circle_segment_area_l458_45877


namespace ali_monday_flowers_l458_45859

/-- The number of flowers Ali sold on Monday -/
def monday_flowers : ℕ := sorry

/-- The number of flowers Ali sold on Tuesday -/
def tuesday_flowers : ℕ := 8

/-- The number of flowers Ali sold on Friday -/
def friday_flowers : ℕ := 2 * monday_flowers

/-- The total number of flowers Ali sold -/
def total_flowers : ℕ := 20

theorem ali_monday_flowers : 
  monday_flowers + tuesday_flowers + friday_flowers = total_flowers → monday_flowers = 4 := by
sorry

end ali_monday_flowers_l458_45859


namespace two_integers_sum_l458_45856

theorem two_integers_sum (x y : ℤ) (h1 : x - y = 1) (h2 : x = -4) (h3 : y = -5) : x + y = -9 := by
  sorry

end two_integers_sum_l458_45856


namespace siblings_average_age_l458_45893

theorem siblings_average_age (youngest_age : ℕ) (age_differences : List ℕ) : 
  youngest_age = 20 → 
  age_differences = [2, 7, 11] →
  (youngest_age + youngest_age + 2 + youngest_age + 7 + youngest_age + 11) / 4 = 25 := by
  sorry

end siblings_average_age_l458_45893


namespace symmetric_difference_of_A_and_B_l458_45883

/-- The set A -/
def A : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}

/-- The set B -/
def B : Set (ℝ × ℝ) := {p | (p.1 - 5) / (p.1 - 4) = 1}

/-- The symmetric difference of two sets -/
def symmetricDifference (X Y : Set α) : Set α :=
  (X \ Y) ∪ (Y \ X)

/-- Theorem: The symmetric difference of A and B -/
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {p : ℝ × ℝ | p.2 = p.1 + 1 ∧ p.1 ≠ 4} := by
  sorry

end symmetric_difference_of_A_and_B_l458_45883


namespace division_problem_l458_45824

theorem division_problem (dividend quotient divisor remainder x : ℕ) : 
  remainder = 5 →
  divisor = 3 * quotient →
  dividend = 113 →
  divisor = 3 * remainder + x →
  dividend = divisor * quotient + remainder →
  x = 3 := by
sorry

end division_problem_l458_45824


namespace rectangle_triangle_area_l458_45810

/-- Given a rectangle ACDE, a point B on AC, a point F on AE, and an equilateral triangle CEF,
    prove that the area of ACDE + CEF - ABF is 1100 + (225 * Real.sqrt 3) / 4 -/
theorem rectangle_triangle_area (A B C D E F : ℝ × ℝ) : 
  let AC : ℝ := 40
  let AE : ℝ := 30
  let AB : ℝ := AC / 3
  let AF : ℝ := AE / 2
  let area_ACDE : ℝ := AC * AE
  let area_CEF : ℝ := (Real.sqrt 3 / 4) * AF^2
  let area_ABF : ℝ := (1 / 2) * AB * AF
  area_ACDE + area_CEF - area_ABF = 1100 + (225 * Real.sqrt 3) / 4 := by
  sorry

end rectangle_triangle_area_l458_45810


namespace balloon_count_l458_45833

/-- Represents the number of balloons of each color and their arrangement --/
structure BalloonArrangement where
  red : Nat
  yellow : Nat
  blue : Nat
  yellow_spaces : Nat
  yellow_unfilled : Nat

/-- Calculates the total number of balloons --/
def total_balloons (arrangement : BalloonArrangement) : Nat :=
  arrangement.red + arrangement.yellow + arrangement.blue

/-- Theorem stating the correct number of yellow and blue balloons --/
theorem balloon_count (arrangement : BalloonArrangement) 
  (h1 : arrangement.red = 40)
  (h2 : arrangement.yellow_spaces = arrangement.red - 1)
  (h3 : arrangement.yellow_unfilled = 3)
  (h4 : arrangement.yellow = arrangement.yellow_spaces + arrangement.yellow_unfilled)
  (h5 : arrangement.blue = total_balloons arrangement - 1) :
  arrangement.yellow = 42 ∧ arrangement.blue = 81 := by
  sorry

#check balloon_count

end balloon_count_l458_45833


namespace exists_alpha_congruence_l458_45818

theorem exists_alpha_congruence : ∃ α : ℤ, α ^ 2 ≡ 2 [ZMOD 7^3] ∧ α ≡ 3 [ZMOD 7] :=
sorry

end exists_alpha_congruence_l458_45818


namespace grass_field_width_l458_45801

/-- Given a rectangular grass field with length 85 m, surrounded by a 2.5 m wide path 
    with an area of 1450 sq m, the width of the grass field is 200 m. -/
theorem grass_field_width (field_length : ℝ) (path_width : ℝ) (path_area : ℝ) :
  field_length = 85 →
  path_width = 2.5 →
  path_area = 1450 →
  ∃ field_width : ℝ,
    (field_length + 2 * path_width) * (field_width + 2 * path_width) -
    field_length * field_width = path_area ∧
    field_width = 200 :=
by sorry

end grass_field_width_l458_45801


namespace bottle_capacity_ratio_l458_45851

theorem bottle_capacity_ratio (c1 c2 : ℝ) : 
  c1 > 0 ∧ c2 > 0 →  -- Capacities are positive
  c1 / 2 + c2 / 4 = (c1 + c2) / 3 →  -- Oil is 1/3 of total mixture
  c2 / c1 = 2 := by sorry

end bottle_capacity_ratio_l458_45851


namespace rotation_of_D_around_E_l458_45821

-- Define the points
def D : ℝ × ℝ := (3, 2)
def E : ℝ × ℝ := (6, 5)
def F : ℝ × ℝ := (6, 2)

-- Define the rotation function
def rotate180AroundPoint (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - point.1, 2 * center.2 - point.2)

-- Theorem statement
theorem rotation_of_D_around_E :
  rotate180AroundPoint E D = (9, 8) := by sorry

end rotation_of_D_around_E_l458_45821


namespace correct_subtraction_result_l458_45815

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Calculates the numeric value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

theorem correct_subtraction_result : ∀ (minuend subtrahend : TwoDigitNumber),
  minuend.units = 3 →
  (minuend.value - 3 + 5) - 25 = 60 →
  subtrahend.value = 52 →
  minuend.value - subtrahend.value = 31 := by
  sorry

end correct_subtraction_result_l458_45815


namespace angle_bisector_ratio_not_unique_l458_45809

/-- Represents a triangle --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ

/-- Represents the ratio of an angle bisector to its corresponding side --/
def angle_bisector_ratio (t : Triangle) : ℝ := 
  sorry -- Definition of angle bisector ratio

/-- Two triangles are similar if their corresponding angles are equal --/
def similar (t1 t2 : Triangle) : Prop :=
  t1.angle_A = t2.angle_A ∧ t1.angle_B = t2.angle_B ∧ t1.angle_C = t2.angle_C

theorem angle_bisector_ratio_not_unique :
  ∃ (t1 t2 : Triangle) (r : ℝ), 
    angle_bisector_ratio t1 = r ∧ 
    angle_bisector_ratio t2 = r ∧ 
    ¬(similar t1 t2) :=
  sorry


end angle_bisector_ratio_not_unique_l458_45809


namespace mode_of_dataset_l458_45872

def dataset : List ℕ := [2, 2, 2, 3, 3, 4]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_dataset : mode dataset = 2 := by
  sorry

end mode_of_dataset_l458_45872


namespace fractional_equation_solution_l458_45819

theorem fractional_equation_solution :
  ∃ x : ℚ, (1 - x) / (2 - x) - 3 = x / (x - 2) ∧ x = 5 / 2 := by
  sorry

end fractional_equation_solution_l458_45819


namespace product_mod_seven_l458_45827

theorem product_mod_seven : (2007 * 2008 * 2009 * 2010) % 7 = 0 := by
  sorry

end product_mod_seven_l458_45827


namespace largest_of_four_consecutive_integers_l458_45849

theorem largest_of_four_consecutive_integers (a b c d : ℕ) : 
  a > 0 ∧ b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ a * b * c * d = 840 → d = 7 := by
  sorry

end largest_of_four_consecutive_integers_l458_45849


namespace constant_ratio_problem_l458_45886

theorem constant_ratio_problem (x y : ℚ) (k : ℚ) : 
  (k = (5 * x - 3) / (y + 20)) → 
  (y = 2 ∧ x = 1 → k = 1/11) → 
  (y = 5 → x = 58/55) := by
  sorry

end constant_ratio_problem_l458_45886


namespace strictly_increasing_f_implies_a_nonneg_l458_45847

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

-- State the theorem
theorem strictly_increasing_f_implies_a_nonneg 
  (h : ∀ x y : ℝ, x < y → f a x < f a y) : 
  a ≥ 0 :=
sorry

end strictly_increasing_f_implies_a_nonneg_l458_45847


namespace target_compound_has_one_iodine_l458_45837

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  nitrogen : ℕ
  hydrogen : ℕ
  iodine : ℕ

/-- Atomic weights of elements -/
def atomic_weight : Fin 3 → ℝ
| 0 => 14.01  -- Nitrogen
| 1 => 1.01   -- Hydrogen
| 2 => 126.90 -- Iodine

/-- Calculate the molecular weight of a compound -/
def molecular_weight (c : Compound) : ℝ :=
  c.nitrogen * atomic_weight 0 + c.hydrogen * atomic_weight 1 + c.iodine * atomic_weight 2

/-- The compound in question -/
def target_compound : Compound := { nitrogen := 1, hydrogen := 4, iodine := 1 }

/-- Theorem stating that the target compound has exactly one iodine atom -/
theorem target_compound_has_one_iodine :
  molecular_weight target_compound = 145 ∧ target_compound.iodine = 1 := by
  sorry

end target_compound_has_one_iodine_l458_45837


namespace triangle_base_height_difference_l458_45867

theorem triangle_base_height_difference (base height : ℚ) : 
  base = 5/6 → height = 4/6 → base - height = 1/6 := by
  sorry

end triangle_base_height_difference_l458_45867


namespace plan_y_more_cost_effective_l458_45860

/-- Cost of Plan X in cents for m megabytes -/
def cost_x (m : ℕ) : ℕ := 15 * m

/-- Cost of Plan Y in cents for m megabytes -/
def cost_y (m : ℕ) : ℕ := 2500 + 7 * m

/-- The minimum whole number of megabytes for Plan Y to be more cost-effective than Plan X -/
def min_megabytes : ℕ := 313

theorem plan_y_more_cost_effective :
  ∀ m : ℕ, m ≥ min_megabytes → cost_y m < cost_x m ∧
  ∀ n : ℕ, n < min_megabytes → cost_y n ≥ cost_x n :=
by sorry

end plan_y_more_cost_effective_l458_45860


namespace determinant_of_roots_l458_45858

theorem determinant_of_roots (s p q : ℝ) (a b c : ℝ) : 
  a^3 + s*a^2 + p*a + q = 0 → 
  b^3 + s*b^2 + p*b + q = 0 → 
  c^3 + s*c^2 + p*c + q = 0 → 
  Matrix.det !![a, b, c; b, c, a; c, a, b] = -s*(s^2 - 3*p) := by
  sorry

end determinant_of_roots_l458_45858


namespace system_solution_unique_l458_45884

theorem system_solution_unique (x y : ℝ) : 
  x + 3 * y = -1 ∧ 2 * x + y = 3 ↔ x = 2 ∧ y = -1 := by
sorry

end system_solution_unique_l458_45884


namespace budget_allocation_circle_graph_l458_45880

theorem budget_allocation_circle_graph (microphotonics : ℝ) (home_electronics : ℝ) 
  (food_additives : ℝ) (genetically_modified_microorganisms : ℝ) (industrial_lubricants : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 10 →
  genetically_modified_microorganisms = 29 →
  industrial_lubricants = 8 →
  (360 : ℝ) * (100 - (microphotonics + home_electronics + food_additives + 
    genetically_modified_microorganisms + industrial_lubricants)) / 100 = 54 := by
  sorry

end budget_allocation_circle_graph_l458_45880


namespace four_distinct_positive_roots_l458_45857

/-- The polynomial f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - x^3 + 8*a*x^2 - a*x + a^2

/-- Theorem stating the condition for f(x) to have four distinct positive roots -/
theorem four_distinct_positive_roots (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ f a x₄ = 0) ↔
  (1/25 < a ∧ a < 1/24) :=
sorry

end four_distinct_positive_roots_l458_45857


namespace increasing_function_a_range_l458_45844

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then (4 - a) * x + 7 else a^x

-- Define what it means for f to be increasing on ℝ
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem increasing_function_a_range :
  ∀ a : ℝ, (is_increasing (f a)) ↔ (3 ≤ a ∧ a < 4) :=
sorry

end increasing_function_a_range_l458_45844


namespace exists_similar_package_with_ten_boxes_l458_45812

/-- Represents a rectangular box with dimensions a, b, and c -/
structure Box where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- Represents a package containing boxes -/
structure Package where
  x : ℝ
  y : ℝ
  z : ℝ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z

/-- Defines geometric similarity between a package and a box -/
def geometricallySimilar (p : Package) (b : Box) : Prop :=
  ∃ k : ℝ, k > 0 ∧ p.x = k * b.a ∧ p.y = k * b.b ∧ p.z = k * b.c

/-- Defines if a package can contain exactly 10 boxes -/
def canContainTenBoxes (p : Package) (b : Box) : Prop :=
  (p.x = 10 * b.a ∧ p.y = b.b ∧ p.z = b.c) ∨
  (p.x = 5 * b.a ∧ p.y = 2 * b.b ∧ p.z = b.c)

/-- Theorem stating that there exists a package geometrically similar to a box and containing 10 boxes -/
theorem exists_similar_package_with_ten_boxes (b : Box) :
  ∃ p : Package, geometricallySimilar p b ∧ canContainTenBoxes p b := by
  sorry

end exists_similar_package_with_ten_boxes_l458_45812


namespace briefcase_pen_price_ratio_l458_45892

/-- Given a pen price of 4 and a total cost of 24 for the pen and a briefcase,
    where the briefcase's price is some multiple of the pen's price,
    prove that the ratio of the briefcase's price to the pen's price is 5. -/
theorem briefcase_pen_price_ratio :
  ∀ (briefcase_price : ℝ),
  briefcase_price > 0 →
  ∃ (multiple : ℝ), multiple > 0 ∧ briefcase_price = 4 * multiple →
  4 + briefcase_price = 24 →
  briefcase_price / 4 = 5 := by
sorry

end briefcase_pen_price_ratio_l458_45892


namespace family_members_count_l458_45876

def num_birds : ℕ := 4
def num_dogs : ℕ := 3
def num_cats : ℕ := 18

def bird_feet : ℕ := 2
def dog_feet : ℕ := 4
def cat_feet : ℕ := 4

def animal_heads : ℕ := num_birds + num_dogs + num_cats

def animal_feet : ℕ := num_birds * bird_feet + num_dogs * dog_feet + num_cats * cat_feet

def human_feet : ℕ := 2
def human_head : ℕ := 1

theorem family_members_count :
  ∃ (F : ℕ), animal_feet + F * human_feet = animal_heads + F * human_head + 74 ∧ F = 7 := by
  sorry

end family_members_count_l458_45876


namespace square_difference_hundred_l458_45870

theorem square_difference_hundred : ∃ x y : ℤ, x^2 - y^2 = 100 ∧ x = 26 ∧ y = 24 := by
  sorry

end square_difference_hundred_l458_45870


namespace fahrenheit_to_celsius_l458_45871

theorem fahrenheit_to_celsius (C F : ℝ) : C = (5/9) * (F - 32) → C = 20 → F = 68 := by
  sorry

end fahrenheit_to_celsius_l458_45871


namespace annie_pays_36_for_12kg_l458_45853

-- Define the price function for oranges
def price (mass : ℝ) : ℝ := sorry

-- Define the given conditions
axiom price_proportional : ∃ k : ℝ, ∀ m : ℝ, price m = k * m
axiom paid_36_for_12kg : price 12 = 36

-- Theorem to prove
theorem annie_pays_36_for_12kg : price 12 = 36 := by
  sorry

end annie_pays_36_for_12kg_l458_45853


namespace two_roots_implies_c_value_l458_45848

/-- A cubic function with a parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

/-- The number of roots of f for a given c -/
def num_roots (c : ℝ) : ℕ := sorry

/-- Theorem stating that if f has exactly two roots, then c is either -2 or 2 -/
theorem two_roots_implies_c_value (c : ℝ) :
  num_roots c = 2 → c = -2 ∨ c = 2 := by sorry

end two_roots_implies_c_value_l458_45848


namespace sum_of_last_two_digits_of_8_pow_2004_l458_45807

/-- The sum of the tens digit and the units digit in the decimal representation of 8^2004 -/
def sum_of_last_two_digits : ℕ :=
  let n : ℕ := 8^2004
  let tens_digit : ℕ := (n / 10) % 10
  let units_digit : ℕ := n % 10
  tens_digit + units_digit

theorem sum_of_last_two_digits_of_8_pow_2004 :
  sum_of_last_two_digits = 5 := by
  sorry

end sum_of_last_two_digits_of_8_pow_2004_l458_45807


namespace walkway_area_is_416_l458_45874

/-- Represents the garden layout and calculates the walkway area -/
def garden_walkway_area (rows : Nat) (cols : Nat) (bed_length : Nat) (bed_width : Nat) (walkway_width : Nat) : Nat :=
  let total_width := cols * bed_length + (cols + 1) * walkway_width
  let total_length := rows * bed_width + (rows + 1) * walkway_width
  let total_area := total_width * total_length
  let bed_area := rows * cols * bed_length * bed_width
  total_area - bed_area

/-- Theorem stating that the walkway area for the given garden configuration is 416 square feet -/
theorem walkway_area_is_416 :
  garden_walkway_area 4 3 8 3 2 = 416 := by
  sorry

end walkway_area_is_416_l458_45874


namespace max_value_constraint_l458_45854

theorem max_value_constraint (x y z : ℝ) (h : 9*x^2 + 4*y^2 + 25*z^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 173 ∧ 
  (∀ a b c : ℝ, 9*a^2 + 4*b^2 + 25*c^2 = 1 → 8*a + 3*b + 10*c ≤ max) ∧
  (8*x + 3*y + 10*z = max) :=
by sorry

end max_value_constraint_l458_45854


namespace total_squares_5x6_l458_45822

/-- The number of squares of a given size in a grid --/
def count_squares (rows : ℕ) (cols : ℕ) (size : ℕ) : ℕ :=
  (rows - size) * (cols - size)

/-- The total number of squares in a 5x6 grid --/
def total_squares : ℕ :=
  count_squares 5 6 1 + count_squares 5 6 2 + count_squares 5 6 3 + count_squares 5 6 4

/-- Theorem: The total number of squares in a 5x6 grid is 40 --/
theorem total_squares_5x6 : total_squares = 40 := by
  sorry

end total_squares_5x6_l458_45822


namespace bingley_bracelets_l458_45861

theorem bingley_bracelets (initial : ℕ) : 
  let kellys_bracelets : ℕ := 16
  let received : ℕ := kellys_bracelets / 4
  let total : ℕ := initial + received
  let given_away : ℕ := total / 3
  let remaining : ℕ := total - given_away
  remaining = 6 → initial = 5 := by sorry

end bingley_bracelets_l458_45861


namespace truck_length_l458_45825

/-- The length of a truck given its speed and tunnel transit time -/
theorem truck_length (tunnel_length : ℝ) (transit_time : ℝ) (speed_mph : ℝ) :
  tunnel_length = 330 →
  transit_time = 6 →
  speed_mph = 45 →
  (speed_mph * 5280 / 3600) * transit_time - tunnel_length = 66 :=
by sorry

end truck_length_l458_45825


namespace greatest_third_term_in_arithmetic_sequence_l458_45868

theorem greatest_third_term_in_arithmetic_sequence :
  ∀ (a d : ℕ),
  a > 0 →
  d > 0 →
  a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 65 →
  (a + 2*d) = 13 ∧ ∀ (b e : ℕ), b > 0 → e > 0 → 
  b + (b + e) + (b + 2*e) + (b + 3*e) + (b + 4*e) = 65 →
  (b + 2*e) ≤ 13 :=
by sorry

end greatest_third_term_in_arithmetic_sequence_l458_45868


namespace symmetry_axis_of_f_l458_45866

/-- The quadratic function f(x) = -2(x-1)^2 + 3 -/
def f (x : ℝ) : ℝ := -2 * (x - 1)^2 + 3

/-- The axis of symmetry for the quadratic function f -/
def axis_of_symmetry : ℝ := 1

/-- Theorem: The axis of symmetry of f(x) = -2(x-1)^2 + 3 is x = 1 -/
theorem symmetry_axis_of_f :
  ∀ x : ℝ, f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
by
  sorry


end symmetry_axis_of_f_l458_45866


namespace original_price_calculation_l458_45829

theorem original_price_calculation (price paid : ℝ) (h1 : paid = 18) (h2 : paid = (1/4) * price) : price = 72 := by
  sorry

end original_price_calculation_l458_45829


namespace g_equals_zero_l458_45838

/-- The function g(x) = 5x - 7 -/
def g (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: g(7/5) = 0 -/
theorem g_equals_zero : g (7 / 5) = 0 := by
  sorry

end g_equals_zero_l458_45838


namespace greatest_q_plus_r_l458_45899

theorem greatest_q_plus_r : ∃ (q r : ℕ+), 
  927 = 21 * q + r ∧ 
  ∀ (q' r' : ℕ+), 927 = 21 * q' + r' → q + r ≥ q' + r' :=
by sorry

end greatest_q_plus_r_l458_45899


namespace egg_distribution_l458_45836

theorem egg_distribution (num_boxes : ℝ) (eggs_per_box : ℝ) (h1 : num_boxes = 2.0) (h2 : eggs_per_box = 1.5) :
  num_boxes * eggs_per_box = 3.0 := by
  sorry

end egg_distribution_l458_45836


namespace fraction_product_l458_45878

theorem fraction_product : (5/8 : ℚ) * (7/9 : ℚ) * (11/13 : ℚ) * (3/5 : ℚ) * (17/19 : ℚ) * (8/15 : ℚ) = 14280/1107000 := by
  sorry

end fraction_product_l458_45878


namespace factorization_identity_l458_45845

theorem factorization_identity (x y : ℝ) : (x - y)^2 + 2*y*(x - y) = (x - y)*(x + y) := by
  sorry

end factorization_identity_l458_45845


namespace modulus_of_one_over_one_plus_i_l458_45890

open Complex

theorem modulus_of_one_over_one_plus_i : 
  let z : ℂ := 1 / (1 + I)
  abs z = Real.sqrt 2 / 2 := by
  sorry

end modulus_of_one_over_one_plus_i_l458_45890


namespace somu_age_relation_somu_age_relation_past_somu_present_age_l458_45894

/-- Somu's present age -/
def somu_age : ℕ := sorry

/-- Somu's father's present age -/
def father_age : ℕ := sorry

/-- Theorem stating the relationship between Somu's age and his father's age -/
theorem somu_age_relation : somu_age = father_age / 3 := by sorry

/-- Theorem stating the relationship between Somu's and his father's ages 10 years ago -/
theorem somu_age_relation_past : somu_age - 10 = (father_age - 10) / 5 := by sorry

/-- Main theorem proving Somu's present age -/
theorem somu_present_age : somu_age = 20 := by sorry

end somu_age_relation_somu_age_relation_past_somu_present_age_l458_45894


namespace system_solution_l458_45811

theorem system_solution :
  ∀ x y : ℕ+,
  (x.val * y.val + x.val + y.val = 71 ∧
   x.val^2 * y.val + x.val * y.val^2 = 880) →
  ((x.val = 11 ∧ y.val = 5) ∨ (x.val = 5 ∧ y.val = 11)) :=
by sorry

end system_solution_l458_45811


namespace peanut_butter_servings_l458_45888

theorem peanut_butter_servings 
  (jar_content : ℚ) 
  (serving_size : ℚ) 
  (h1 : jar_content = 35 + 4/5)
  (h2 : serving_size = 5/2) : 
  jar_content / serving_size = 14 + 8/25 := by
sorry

end peanut_butter_servings_l458_45888


namespace circle_properties_l458_45889

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the line equation for the center
def center_line (x y : ℝ) : Prop := y = -2 * x

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

theorem circle_properties :
  -- The circle passes through point A(2, -1)
  circle_equation 2 (-1) ∧
  -- The center of the circle is on the line y = -2x
  ∃ (cx cy : ℝ), center_line cx cy ∧ 
    ∀ (x y : ℝ), circle_equation x y ↔ (x - cx)^2 + (y - cy)^2 = 2 ∧
  -- The circle is tangent to the line x + y - 1 = 0
  ∃ (tx ty : ℝ), tangent_line tx ty ∧
    circle_equation tx ty ∧
    ∀ (x y : ℝ), tangent_line x y → 
      ((x - tx)^2 + (y - ty)^2 < 2 ∨ (x = tx ∧ y = ty))
  := by sorry

end circle_properties_l458_45889


namespace square_area_l458_45816

theorem square_area (x : ℝ) : 
  (6 * x - 18 = 3 * x + 9) → 
  (6 * x - 18)^2 = 1296 := by
sorry

end square_area_l458_45816


namespace min_disks_is_fifteen_l458_45820

/-- Represents the storage problem with given file sizes and disk capacity. -/
structure StorageProblem where
  total_files : ℕ
  disk_capacity : ℚ
  files_09mb : ℕ
  files_08mb : ℕ
  files_05mb : ℕ
  h_total : total_files = files_09mb + files_08mb + files_05mb

/-- Calculates the minimum number of disks required for the given storage problem. -/
def min_disks_required (p : StorageProblem) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of disks required for the given problem is 15. -/
theorem min_disks_is_fifteen :
  let p : StorageProblem := {
    total_files := 35,
    disk_capacity := 8/5,
    files_09mb := 5,
    files_08mb := 10,
    files_05mb := 20,
    h_total := by rfl
  }
  min_disks_required p = 15 := by
  sorry

end min_disks_is_fifteen_l458_45820


namespace clock_overlaps_in_24_hours_l458_45835

/-- Represents a clock with an hour hand and a minute hand -/
structure Clock :=
  (hour_revolutions : ℕ)
  (minute_revolutions : ℕ)

/-- The number of overlaps between the hour and minute hands -/
def overlaps (c : Clock) : ℕ := c.minute_revolutions - c.hour_revolutions

theorem clock_overlaps_in_24_hours :
  ∃ (c : Clock), c.hour_revolutions = 2 ∧ c.minute_revolutions = 24 ∧ overlaps c = 22 :=
sorry

end clock_overlaps_in_24_hours_l458_45835


namespace vector_angle_problem_l458_45898

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_angle_problem (b : ℝ × ℝ) :
  let a : ℝ × ℝ := (1, Real.sqrt 3)
  (Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 1) →
  (Real.sqrt (((a.1 + b.1) ^ 2) + ((a.2 + b.2) ^ 2)) = Real.sqrt 3) →
  angle_between_vectors a b = (2 * Real.pi) / 3 := by
  sorry

end vector_angle_problem_l458_45898


namespace fraction_sqrt_cube_root_equals_power_l458_45834

theorem fraction_sqrt_cube_root_equals_power (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 * b)^(1/2) / (a * b)^(1/3) = a^(7/6) * b^(1/6) := by sorry

end fraction_sqrt_cube_root_equals_power_l458_45834


namespace sqrt_of_four_l458_45832

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem statement
theorem sqrt_of_four : sqrt 4 = {2, -2} := by sorry

end sqrt_of_four_l458_45832


namespace solution_set_correct_l458_45863

/-- The solution set of the equation 3sin(x) = 1 + cos(2x) -/
def SolutionSet : Set ℝ :=
  {x | ∃ k : ℤ, x = k * Real.pi + (-1)^k * (Real.pi / 6)}

/-- The original equation -/
def OriginalEquation (x : ℝ) : Prop :=
  3 * Real.sin x = 1 + Real.cos (2 * x)

theorem solution_set_correct :
  ∀ x : ℝ, x ∈ SolutionSet ↔ OriginalEquation x := by
  sorry

end solution_set_correct_l458_45863


namespace movie_screening_attendance_l458_45831

theorem movie_screening_attendance (total_guests : ℕ) 
  (h1 : total_guests = 50)
  (h2 : ∃ women : ℕ, women = total_guests / 2)
  (h3 : ∃ men : ℕ, men = 15)
  (h4 : ∃ children : ℕ, children = total_guests - (total_guests / 2 + 15))
  (h5 : ∃ men_left : ℕ, men_left = 15 / 5)
  (h6 : ∃ children_left : ℕ, children_left = 4) :
  total_guests - (15 / 5 + 4) = 43 := by
sorry


end movie_screening_attendance_l458_45831


namespace fermat_for_small_exponents_l458_45865

theorem fermat_for_small_exponents (x y z n : ℕ) (h : n ≥ z) :
  x^n + y^n ≠ z^n := by
  sorry

end fermat_for_small_exponents_l458_45865


namespace sum_of_complex_roots_l458_45881

theorem sum_of_complex_roots (a b c : ℂ) 
  (eq1 : a^2 = b - c) 
  (eq2 : b^2 = c - a) 
  (eq3 : c^2 = a - b) : 
  (a + b + c = 0) ∨ (a + b + c = Complex.I * Real.sqrt 6) ∨ (a + b + c = -Complex.I * Real.sqrt 6) := by
  sorry

end sum_of_complex_roots_l458_45881


namespace pharmaceutical_royalties_l458_45843

theorem pharmaceutical_royalties (first_royalties second_royalties second_sales : ℝ)
  (ratio_decrease : ℝ) (h1 : first_royalties = 8)
  (h2 : second_royalties = 9) (h3 : second_sales = 108)
  (h4 : ratio_decrease = 0.7916666666666667) :
  ∃ first_sales : ℝ,
    first_sales = 20 ∧
    (first_royalties / first_sales) - (second_royalties / second_sales) =
      ratio_decrease * (first_royalties / first_sales) :=
by sorry

end pharmaceutical_royalties_l458_45843


namespace complex_modulus_l458_45895

theorem complex_modulus (z : ℂ) : z * (1 + Complex.I) = Complex.I → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_l458_45895


namespace smallest_factorizable_b_l458_45839

/-- Represents a factorization of x^2 + bx + 2016 into (x + r)(x + s) -/
structure Factorization where
  r : ℤ
  s : ℤ
  sum_eq : r + s = b
  product_eq : r * s = 2016

/-- Returns true if the quadratic x^2 + bx + 2016 can be factored with integer coefficients -/
def has_integer_factorization (b : ℤ) : Prop :=
  ∃ f : Factorization, f.r + f.s = b ∧ f.r * f.s = 2016

theorem smallest_factorizable_b :
  (has_integer_factorization 90) ∧
  (∀ b : ℤ, 0 < b → b < 90 → ¬(has_integer_factorization b)) :=
sorry

end smallest_factorizable_b_l458_45839


namespace no_rational_roots_for_all_quadratics_l458_45879

/-- The largest known prime number -/
def p : ℕ := 2^24036583 - 1

/-- Theorem stating that there are no positive integers c such that
    both p^2 - 4c and p^2 + 4c are perfect squares -/
theorem no_rational_roots_for_all_quadratics :
  ¬∃ c : ℕ+, ∃ a b : ℕ, (p^2 - 4*c.val = a^2) ∧ (p^2 + 4*c.val = b^2) :=
sorry

end no_rational_roots_for_all_quadratics_l458_45879


namespace cookie_boxes_problem_l458_45808

theorem cookie_boxes_problem (n : ℕ) : 
  (n ≥ 1) →
  (n - 7 ≥ 1) →
  (n - 2 ≥ 1) →
  ((n - 7) + (n - 2) < n) →
  (n = 8) := by
  sorry

end cookie_boxes_problem_l458_45808


namespace least_subtraction_for_divisibility_l458_45875

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(14 ∣ (427398 - y))) ∧ 
  (14 ∣ (427398 - x)) := by
  sorry

end least_subtraction_for_divisibility_l458_45875


namespace triangle_area_triangle_area_proof_l458_45804

/-- The area of the triangle formed by points (0, 0), (4, 2), and (4, -4) is 4√5 square units -/
theorem triangle_area : ℝ :=
let A : ℝ × ℝ := (0, 0)
let B : ℝ × ℝ := (4, 2)
let C : ℝ × ℝ := (4, -4)
let triangle_area := Real.sqrt 5 * 4
triangle_area

/-- Proof that the area of the triangle formed by points (0, 0), (4, 2), and (4, -4) is 4√5 square units -/
theorem triangle_area_proof : triangle_area = Real.sqrt 5 * 4 := by
  sorry

end triangle_area_triangle_area_proof_l458_45804


namespace unique_arrangement_l458_45823

-- Define the types for containers and liquids
inductive Container : Type
  | Bottle
  | Glass
  | Jug
  | Jar

inductive Liquid : Type
  | Milk
  | Lemonade
  | Kvass
  | Water

-- Define the arrangement as a function from Container to Liquid
def Arrangement := Container → Liquid

-- Define the conditions
def water_milk_not_in_bottle (arr : Arrangement) : Prop :=
  arr Container.Bottle ≠ Liquid.Water ∧ arr Container.Bottle ≠ Liquid.Milk

def lemonade_between_jug_and_kvass (arr : Arrangement) : Prop :=
  (arr Container.Bottle = Liquid.Lemonade ∧ arr Container.Jug = Liquid.Milk ∧ arr Container.Jar = Liquid.Kvass) ∨
  (arr Container.Glass = Liquid.Lemonade ∧ arr Container.Bottle = Liquid.Kvass ∧ arr Container.Jar = Liquid.Milk) ∨
  (arr Container.Bottle = Liquid.Milk ∧ arr Container.Glass = Liquid.Lemonade ∧ arr Container.Jug = Liquid.Kvass)

def jar_not_lemonade_or_water (arr : Arrangement) : Prop :=
  arr Container.Jar ≠ Liquid.Lemonade ∧ arr Container.Jar ≠ Liquid.Water

def glass_next_to_jar_and_milk (arr : Arrangement) : Prop :=
  (arr Container.Glass = Liquid.Water ∧ arr Container.Jug = Liquid.Milk) ∨
  (arr Container.Glass = Liquid.Kvass ∧ arr Container.Bottle = Liquid.Milk)

-- Define the correct arrangement
def correct_arrangement : Arrangement :=
  fun c => match c with
  | Container.Bottle => Liquid.Lemonade
  | Container.Glass => Liquid.Water
  | Container.Jug => Liquid.Milk
  | Container.Jar => Liquid.Kvass

-- Theorem statement
theorem unique_arrangement :
  ∀ (arr : Arrangement),
    water_milk_not_in_bottle arr ∧
    lemonade_between_jug_and_kvass arr ∧
    jar_not_lemonade_or_water arr ∧
    glass_next_to_jar_and_milk arr →
    arr = correct_arrangement :=
by sorry

end unique_arrangement_l458_45823


namespace equation_solution_l458_45897

theorem equation_solution : 
  ∃ x : ℝ, (3 / (2 * x + 1) = 5 / (4 * x)) ∧ x = 5/2 := by
  sorry

end equation_solution_l458_45897


namespace remaining_cube_volume_l458_45850

/-- Given a cube with edge length 3 cm, if we cut out 6 smaller cubes each with edge length 1 cm,
    the remaining volume is 21 cm³. -/
theorem remaining_cube_volume :
  let large_cube_edge : ℝ := 3
  let small_cube_edge : ℝ := 1
  let num_faces : ℕ := 6
  let original_volume := large_cube_edge ^ 3
  let cut_out_volume := num_faces * small_cube_edge ^ 3
  original_volume - cut_out_volume = 21 := by sorry

end remaining_cube_volume_l458_45850


namespace line_slope_l458_45882

theorem line_slope (m n p K : ℝ) (h1 : p = 0.3333333333333333) : 
  (m = K * n + 5 ∧ m + 2 = K * (n + p) + 5) → K = 6 := by
  sorry

end line_slope_l458_45882


namespace f_monotone_increasing_l458_45841

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2

theorem f_monotone_increasing :
  (∀ x y, x < y ∧ y < -1 → f x < f y) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
sorry

end f_monotone_increasing_l458_45841


namespace triangle_sine_inequality_l458_45830

theorem triangle_sine_inequality (A B C : Real) (h_triangle : A + B + C = Real.pi) :
  -2 < Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ∧
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ≤ (3 / 2) * Real.sqrt 3 ∧
  (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = (3 / 2) * Real.sqrt 3 ↔
   A = 7 * Real.pi / 9 ∧ B = Real.pi / 9 ∧ C = Real.pi / 9) :=
by sorry

end triangle_sine_inequality_l458_45830


namespace boys_to_girls_ratio_l458_45817

theorem boys_to_girls_ratio (S G : ℚ) (h : S > 0) (h1 : G > 0) (h2 : (1/2) * G = (1/3) * S) :
  (S - G) / G = 1/2 := by
  sorry

end boys_to_girls_ratio_l458_45817


namespace fraction_equality_l458_45803

theorem fraction_equality (a b : ℝ) (h : a / b = 3 / 4) : (a - b) / (a + b) = -1 / 7 := by
  sorry

end fraction_equality_l458_45803


namespace opposite_of_negative_2023_l458_45873

theorem opposite_of_negative_2023 : 
  (-((-2023 : ℝ)) = (2023 : ℝ)) := by sorry

end opposite_of_negative_2023_l458_45873


namespace vacation_cost_theorem_l458_45862

/-- Calculates the total cost of a vacation in USD given specific expenses and exchange rates -/
def vacation_cost (num_people : ℕ) 
                  (rent_per_person : ℝ) 
                  (transport_per_person : ℝ) 
                  (food_per_person : ℝ) 
                  (activities_per_person : ℝ) 
                  (euro_to_usd : ℝ) 
                  (pound_to_usd : ℝ) 
                  (yen_to_usd : ℝ) : ℝ :=
  let total_rent := num_people * rent_per_person * euro_to_usd
  let total_transport := num_people * transport_per_person
  let total_food := num_people * food_per_person * pound_to_usd
  let total_activities := num_people * activities_per_person * yen_to_usd
  total_rent + total_transport + total_food + total_activities

/-- The total cost of the vacation is $1384.25 -/
theorem vacation_cost_theorem : 
  vacation_cost 7 65 25 50 2750 1.2 1.4 0.009 = 1384.25 := by
  sorry

end vacation_cost_theorem_l458_45862


namespace sum_and_round_l458_45864

def round_to_nearest_hundred (x : ℤ) : ℤ :=
  100 * ((x + 50) / 100)

theorem sum_and_round : round_to_nearest_hundred (128 + 264) = 400 := by
  sorry

end sum_and_round_l458_45864


namespace product_of_numbers_l458_45885

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 30) (sum_cubes_eq : x^3 + y^3 = 9450) : x * y = -585 := by
  sorry

end product_of_numbers_l458_45885


namespace one_positive_integer_solution_l458_45800

theorem one_positive_integer_solution : 
  ∃! (n : ℕ), n > 0 ∧ (25 : ℝ) - 5 * n > 15 :=
by sorry

end one_positive_integer_solution_l458_45800


namespace profit_share_difference_l458_45813

/-- Given the investments and profit share of B, calculate the difference between profit shares of A and C -/
theorem profit_share_difference (investment_A investment_B investment_C profit_B : ℕ) : 
  investment_A = 8000 →
  investment_B = 10000 →
  investment_C = 12000 →
  profit_B = 1900 →
  ∃ (profit_A profit_C : ℕ),
    profit_A * investment_B = profit_B * investment_A ∧
    profit_C * investment_B = profit_B * investment_C ∧
    profit_C - profit_A = 760 :=
by sorry

end profit_share_difference_l458_45813


namespace last_two_digits_of_sum_l458_45896

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_series : ℕ → ℕ
  | 0 => 0
  | n + 1 => sum_series n + if (n + 1) % 3 = 0 ∧ n + 1 ≤ 9 then 2 * factorial (n + 1) else 0

theorem last_two_digits_of_sum : last_two_digits (sum_series 99) = 24 := by
  sorry

end last_two_digits_of_sum_l458_45896


namespace smallest_y_in_arithmetic_series_l458_45846

theorem smallest_y_in_arithmetic_series (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →  -- all terms are positive
  ∃ d : ℝ, x = y - d ∧ z = y + d →  -- arithmetic series condition
  x * y * z = 125 →  -- product condition
  y ≥ 5 ∧ ∀ y' : ℝ, (∃ x' z' : ℝ, x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ 
    (∃ d' : ℝ, x' = y' - d' ∧ z' = y' + d') ∧ 
    x' * y' * z' = 125) → y' ≥ 5 := by
  sorry

#check smallest_y_in_arithmetic_series

end smallest_y_in_arithmetic_series_l458_45846


namespace arithmetic_sequence_sum_l458_45891

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 = 2 →                    -- a_1 = 2
  a 2 + a 3 = 13 →             -- a_2 + a_3 = 13
  a 4 + a 5 + a 6 = 42 :=      -- conclusion: a_4 + a_5 + a_6 = 42
by
  sorry

end arithmetic_sequence_sum_l458_45891


namespace circle_max_min_distances_l458_45814

/-- Circle C with center (3,4) and radius 1 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 1}

/-- Point A -/
def A : ℝ × ℝ := (-1, 0)

/-- Point B -/
def B : ℝ × ℝ := (1, 0)

/-- Distance squared between two points -/
def distanceSquared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- The expression to be maximized and minimized -/
def d (P : ℝ × ℝ) : ℝ :=
  distanceSquared P A + distanceSquared P B

theorem circle_max_min_distances :
  (∀ P ∈ C, d P ≤ 74) ∧ (∀ P ∈ C, d P ≥ 34) ∧ (∃ P ∈ C, d P = 74) ∧ (∃ P ∈ C, d P = 34) := by
  sorry

end circle_max_min_distances_l458_45814


namespace periodic_function_l458_45887

theorem periodic_function (f : ℝ → ℝ) (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)) →
  ∀ x : ℝ, f x = f (x + 2*a) :=
by sorry

end periodic_function_l458_45887


namespace no_number_with_2011_quotient_and_remainder_l458_45840

-- Function to calculate the sum of digits of a natural number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_number_with_2011_quotient_and_remainder :
  ¬ ∃ (n : ℕ), 
    let s := sumOfDigits n
    n / s = 2011 ∧ n % s = 2011 := by
  sorry

end no_number_with_2011_quotient_and_remainder_l458_45840


namespace absolute_value_equation_solution_count_l458_45826

theorem absolute_value_equation_solution_count : 
  ∃! x : ℝ, |x - 5| = |x + 3| := by sorry

end absolute_value_equation_solution_count_l458_45826


namespace taxi_fare_problem_l458_45806

/-- Represents the fare for a taxi ride. -/
structure TaxiFare where
  distance : ℝ  -- Distance traveled in kilometers
  cost : ℝ      -- Cost in dollars
  h_positive : distance > 0

/-- States that taxi fares are directly proportional to the distance traveled. -/
def DirectlyProportional (f₁ f₂ : TaxiFare) : Prop :=
  f₁.cost / f₁.distance = f₂.cost / f₂.distance

theorem taxi_fare_problem (f₁ : TaxiFare) 
    (h₁ : f₁.distance = 80 ∧ f₁.cost = 200) :
    ∃ (f₂ : TaxiFare), 
      f₂.distance = 120 ∧ 
      DirectlyProportional f₁ f₂ ∧ 
      f₂.cost = 300 := by
  sorry

end taxi_fare_problem_l458_45806
