import Mathlib

namespace freddy_age_l1404_140433

theorem freddy_age (F M R : ℕ) 
  (sum_ages : F + M + R = 35)
  (matthew_rebecca : M = R + 2)
  (freddy_matthew : F = M + 4) :
  F = 15 := by
sorry

end freddy_age_l1404_140433


namespace average_temperature_proof_l1404_140412

/-- Given the average temperature for four days and individual temperatures for two days,
    prove that the average temperature for a different set of four days is as calculated. -/
theorem average_temperature_proof
  (avg_mon_to_thu : ℝ)
  (temp_mon : ℝ)
  (temp_fri : ℝ)
  (h1 : avg_mon_to_thu = 48)
  (h2 : temp_mon = 40)
  (h3 : temp_fri = 32) :
  (4 * avg_mon_to_thu - temp_mon + temp_fri) / 4 = 46 := by
  sorry


end average_temperature_proof_l1404_140412


namespace student_ticket_cost_l1404_140430

theorem student_ticket_cost (num_students : ℕ) (num_teachers : ℕ) (adult_ticket_cost : ℚ) (total_cost : ℚ) :
  num_students = 12 →
  num_teachers = 4 →
  adult_ticket_cost = 3 →
  total_cost = 24 →
  ∃ (student_ticket_cost : ℚ),
    student_ticket_cost * num_students + adult_ticket_cost * num_teachers = total_cost ∧
    student_ticket_cost = 1 :=
by sorry

end student_ticket_cost_l1404_140430


namespace perfect_square_trinomial_l1404_140453

/-- If 4x^2 - (a-b)x + 9 is a perfect square trinomial, then 2a-2b = ±24 -/
theorem perfect_square_trinomial (a b : ℝ) :
  (∃ c : ℝ, ∀ x : ℝ, 4*x^2 - (a-b)*x + 9 = (2*x - c)^2) →
  (2*a - 2*b = 24 ∨ 2*a - 2*b = -24) :=
by sorry

end perfect_square_trinomial_l1404_140453


namespace negation_equivalence_l1404_140484

-- Define the universe of discourse
variable (Student : Type)

-- Define the property of being patient
variable (isPatient : Student → Prop)

-- Statement (6): All students are patient
def allStudentsPatient : Prop := ∀ s : Student, isPatient s

-- Statement (5): At least one student is impatient
def oneStudentImpatient : Prop := ∃ s : Student, ¬(isPatient s)

-- Theorem: Statement (5) is equivalent to the negation of statement (6)
theorem negation_equivalence : oneStudentImpatient Student isPatient ↔ ¬(allStudentsPatient Student isPatient) := by
  sorry

end negation_equivalence_l1404_140484


namespace max_difference_intersection_points_l1404_140475

/-- The first function f(x) = 2 - x^2 + 2x^3 -/
def f (x : ℝ) : ℝ := 2 - x^2 + 2*x^3

/-- The second function g(x) = 3 + 2x^2 + 2x^3 -/
def g (x : ℝ) : ℝ := 3 + 2*x^2 + 2*x^3

/-- Theorem stating that the maximum difference between y-coordinates of intersection points is 4√3/9 -/
theorem max_difference_intersection_points :
  ∃ (x₁ x₂ : ℝ), f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ 
  ∀ (y₁ y₂ : ℝ), (∃ (x : ℝ), f x = g x ∧ (y₁ = f x ∨ y₁ = g x)) →
                 (∃ (x : ℝ), f x = g x ∧ (y₂ = f x ∨ y₂ = g x)) →
                 |y₁ - y₂| ≤ 4 * Real.sqrt 3 / 9 :=
by sorry

end max_difference_intersection_points_l1404_140475


namespace max_common_ratio_arithmetic_geometric_l1404_140472

theorem max_common_ratio_arithmetic_geometric (a : ℕ → ℝ) (d q : ℝ) (k : ℕ) :
  (∀ n, a (n + 1) - a n = d) →  -- arithmetic sequence condition
  d ≠ 0 →  -- non-zero common difference
  k ≥ 2 →  -- k condition
  a k / a 1 = q →  -- geometric sequence condition for a_1 and a_k
  a (2 * k) / a k = q →  -- geometric sequence condition for a_k and a_2k
  q ≤ 2 :=
by sorry

end max_common_ratio_arithmetic_geometric_l1404_140472


namespace patio_length_l1404_140460

theorem patio_length (w l : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 100) : l = 40 := by
  sorry

end patio_length_l1404_140460


namespace complex_magnitude_equals_five_sqrt_five_l1404_140476

theorem complex_magnitude_equals_five_sqrt_five (t : ℝ) :
  t > 0 → (Complex.abs (-5 + t * Complex.I) = 5 * Real.sqrt 5 ↔ t = 10) := by
sorry

end complex_magnitude_equals_five_sqrt_five_l1404_140476


namespace waiter_customers_l1404_140406

/-- Given a waiter with tables, each having a certain number of women and men,
    calculate the total number of customers. -/
theorem waiter_customers (tables women_per_table men_per_table : ℕ) :
  tables = 5 →
  women_per_table = 5 →
  men_per_table = 3 →
  tables * (women_per_table + men_per_table) = 40 := by
sorry

end waiter_customers_l1404_140406


namespace value_of_3b_plus_4c_l1404_140417

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 2

-- Define the function f
def f (b c x : ℝ) : ℝ := b * x + c

-- State the theorem
theorem value_of_3b_plus_4c (b c : ℝ) :
  (∃ f_inv : ℝ → ℝ, 
    (∀ x, f b c (f_inv x) = x ∧ f_inv (f b c x) = x) ∧ 
    (∀ x, g x = 2 * f_inv x + 4)) →
  3 * b + 4 * c = 14/3 :=
by sorry

end value_of_3b_plus_4c_l1404_140417


namespace min_people_theorem_l1404_140444

/-- Represents a group of people consisting of married couples -/
structure CoupleGroup :=
  (num_couples : ℕ)
  (total_people : ℕ)
  (h_total : total_people = 2 * num_couples)

/-- The minimum number of people required to guarantee at least one married couple -/
def min_for_couple (group : CoupleGroup) : ℕ :=
  group.num_couples + 3

/-- The minimum number of people required to guarantee at least two people of the same gender -/
def min_for_same_gender (group : CoupleGroup) : ℕ := 3

/-- Theorem stating the minimum number of people required for both conditions -/
theorem min_people_theorem (group : CoupleGroup) 
  (h_group : group.num_couples = 10) : 
  min_for_couple group = 13 ∧ min_for_same_gender group = 3 := by
  sorry

#eval min_for_couple ⟨10, 20, rfl⟩
#eval min_for_same_gender ⟨10, 20, rfl⟩

end min_people_theorem_l1404_140444


namespace total_campers_rowing_l1404_140447

theorem total_campers_rowing (morning_campers afternoon_campers : ℕ) 
  (h1 : morning_campers = 35) 
  (h2 : afternoon_campers = 27) : 
  morning_campers + afternoon_campers = 62 := by
  sorry

end total_campers_rowing_l1404_140447


namespace second_term_of_geometric_series_l1404_140468

theorem second_term_of_geometric_series 
  (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = (1 : ℝ) / 4)
  (h2 : S = 48) (h3 : S = a / (1 - r)) :
  a * r = 9 := by
sorry

end second_term_of_geometric_series_l1404_140468


namespace intersection_range_l1404_140437

/-- The function f(x) = x³ - 3x - 1 --/
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

/-- Theorem: If the line y = m intersects the graph of f(x) = x³ - 3x - 1
    at three distinct points, then m is in the open interval (-3, 1) --/
theorem intersection_range (m : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) →
  m > -3 ∧ m < 1 :=
sorry

end intersection_range_l1404_140437


namespace unique_a_value_l1404_140454

/-- Converts a number from base 53 to base 10 -/
def base53ToBase10 (n : ℕ) : ℕ := sorry

/-- Theorem: If a is an integer between 0 and 20 (inclusive) and 4254253₅₃ - a is a multiple of 17, then a = 3 -/
theorem unique_a_value (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 20) 
  (h3 : (base53ToBase10 4254253 - a) % 17 = 0) : a = 3 := by sorry

end unique_a_value_l1404_140454


namespace sin_arccos_eight_seventeenths_l1404_140413

theorem sin_arccos_eight_seventeenths : 
  Real.sin (Real.arccos (8 / 17)) = 15 / 17 := by
  sorry

end sin_arccos_eight_seventeenths_l1404_140413


namespace village_b_largest_population_l1404_140467

/-- Calculate the population after n years given initial population and growth rate -/
def futurePopulation (initialPop : ℝ) (growthRate : ℝ) (years : ℕ) : ℝ :=
  initialPop * (1 + growthRate) ^ years

/-- Theorem: Village B has the largest population after 3 years -/
theorem village_b_largest_population :
  let villageA := futurePopulation 12000 0.24 3
  let villageB := futurePopulation 15000 0.18 3
  let villageC := futurePopulation 18000 (-0.12) 3
  villageB > villageA ∧ villageB > villageC := by sorry

end village_b_largest_population_l1404_140467


namespace maximize_x_cubed_y_fifth_l1404_140404

theorem maximize_x_cubed_y_fifth (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  x^3 * y^5 ≤ (7.5^3) * (12.5^5) ∧ 
  (x^3 * y^5 = (7.5^3) * (12.5^5) ↔ x = 7.5 ∧ y = 12.5) := by
sorry

end maximize_x_cubed_y_fifth_l1404_140404


namespace sin_ten_degrees_root_l1404_140436

theorem sin_ten_degrees_root : ∃ x : ℝ, 
  (x = Real.sin (10 * π / 180)) ∧ 
  (8 * x^3 - 6 * x + 1 = 0) := by
  sorry

end sin_ten_degrees_root_l1404_140436


namespace product_of_sums_powers_l1404_140450

theorem product_of_sums_powers : (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) = 63403380965376 := by
  sorry

end product_of_sums_powers_l1404_140450


namespace remainder_problem_l1404_140431

theorem remainder_problem (N : ℤ) : 
  ∃ k : ℤ, N = 761 * k + 173 → N % 29 = 28 := by
  sorry

end remainder_problem_l1404_140431


namespace plane_equation_l1404_140483

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a plane -/
def pointOnPlane (pt : Point3D) (pl : Plane) : Prop :=
  pl.a * pt.x + pl.b * pt.y + pl.c * pt.z + pl.d = 0

/-- Check if two planes are perpendicular -/
def planesArePerpendicular (pl1 pl2 : Plane) : Prop :=
  pl1.a * pl2.a + pl1.b * pl2.b + pl1.c * pl2.c = 0

/-- The main theorem -/
theorem plane_equation : ∃ (pl : Plane),
  pointOnPlane ⟨0, 2, 1⟩ pl ∧
  pointOnPlane ⟨2, 0, 1⟩ pl ∧
  planesArePerpendicular pl ⟨2, -1, 3, -4⟩ ∧
  pl.a > 0 ∧
  Int.gcd (Int.natAbs (Int.floor pl.a)) (Int.gcd (Int.natAbs (Int.floor pl.b)) (Int.gcd (Int.natAbs (Int.floor pl.c)) (Int.natAbs (Int.floor pl.d)))) = 1 ∧
  pl = ⟨1, 1, -1, -1⟩ :=
by
  sorry

end plane_equation_l1404_140483


namespace cubic_system_product_l1404_140470

theorem cubic_system_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2010 ∧ y₁^3 - 3*x₁^2*y₁ = 2000)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2010 ∧ y₂^3 - 3*x₂^2*y₂ = 2000)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2010 ∧ y₃^3 - 3*x₃^2*y₃ = 2000) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/100 := by
sorry

end cubic_system_product_l1404_140470


namespace symmetric_line_across_x_axis_l1404_140407

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflects a line across the x-axis -/
def reflectLineAcrossXAxis (l : Line2D) : Line2D :=
  { a := l.a, b := -l.b, c := l.c }

theorem symmetric_line_across_x_axis :
  let originalLine := Line2D.mk 2 (-3) 2
  let symmetricLine := Line2D.mk 2 3 2
  reflectLineAcrossXAxis originalLine = symmetricLine := by sorry

end symmetric_line_across_x_axis_l1404_140407


namespace lucky_set_guaranteed_l1404_140492

/-- The number of cards in the deck -/
def deck_size : ℕ := 52

/-- The maximum sum of digits possible for any card in the deck -/
def max_sum : ℕ := 13

/-- Function to calculate the sum of digits for a given number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Theorem stating that drawing 26 cards guarantees a "lucky" set -/
theorem lucky_set_guaranteed (drawn : ℕ) (h : drawn = 26) :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a ≤ deck_size ∧ b ≤ deck_size ∧ c ≤ deck_size ∧
  sum_of_digits a = sum_of_digits b ∧ sum_of_digits b = sum_of_digits c :=
by sorry

end lucky_set_guaranteed_l1404_140492


namespace password_recovery_l1404_140420

def alphabet_size : Nat := 32

def encode (c : Char) : Nat := 
  sorry

def decode (n : Nat) : Char := 
  sorry

def generate_x (a b x : Nat) : Nat :=
  (a * x + b) % 10

def generate_c (x y : Nat) : Nat :=
  (x + y) % 10

def is_valid_sequence (s : List Nat) (password : String) (a b : Nat) : Prop :=
  sorry

theorem password_recovery (a b : Nat) : 
  ∃ (password : String),
    password.length = 4 ∧ 
    is_valid_sequence [2, 8, 5, 2, 8, 3, 1, 9, 8, 4, 1, 8, 4, 9, 7] (password ++ password) a b ∧
    password = "яхта" :=
  sorry

end password_recovery_l1404_140420


namespace melted_ice_cream_height_l1404_140424

/-- Given a sphere of ice cream with radius 3 inches that melts into a cylinder
    with radius 12 inches while maintaining constant density, the height of the
    resulting cylinder is 1/4 inch. -/
theorem melted_ice_cream_height (r_sphere r_cylinder : ℝ) (h : ℝ) : 
  r_sphere = 3 →
  r_cylinder = 12 →
  (4 / 3) * π * r_sphere^3 = π * r_cylinder^2 * h →
  h = 1 / 4 := by sorry

end melted_ice_cream_height_l1404_140424


namespace negative_cube_squared_l1404_140435

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end negative_cube_squared_l1404_140435


namespace greatest_common_divisor_540_462_l1404_140498

theorem greatest_common_divisor_540_462 : Nat.gcd 540 462 = 6 := by
  sorry

end greatest_common_divisor_540_462_l1404_140498


namespace not_always_int_greater_than_decimal_l1404_140414

-- Define a decimal as a structure with an integer part and a fractional part
structure Decimal where
  integerPart : Int
  fractionalPart : Rat
  fractionalPart_lt_one : fractionalPart < 1

-- Define the comparison between an integer and a decimal
def intGreaterThanDecimal (n : Int) (d : Decimal) : Prop :=
  n > d.integerPart + d.fractionalPart

-- Theorem statement
theorem not_always_int_greater_than_decimal :
  ¬ ∀ (n : Int) (d : Decimal), intGreaterThanDecimal n d :=
sorry

end not_always_int_greater_than_decimal_l1404_140414


namespace smallest_greater_than_1_1_l1404_140456

def S : Set ℚ := {1.4, 9/10, 1.2, 0.5, 13/10}

theorem smallest_greater_than_1_1 : 
  ∃ x ∈ S, x > 1.1 ∧ ∀ y ∈ S, y > 1.1 → x ≤ y :=
by sorry

end smallest_greater_than_1_1_l1404_140456


namespace train_speed_calculation_l1404_140481

/-- Proves that given the specified conditions, the train's speed is 36 kmph -/
theorem train_speed_calculation (jogger_speed : ℝ) (jogger_ahead : ℝ) (train_length : ℝ) (pass_time : ℝ) :
  jogger_speed = 9 →
  jogger_ahead = 240 →
  train_length = 120 →
  pass_time = 35.99712023038157 →
  (jogger_ahead + train_length) / pass_time * 3.6 = 36 := by
  sorry

#eval (240 + 120) / 35.99712023038157 * 3.6

end train_speed_calculation_l1404_140481


namespace square_nine_implies_fourth_power_eightyone_l1404_140464

theorem square_nine_implies_fourth_power_eightyone (a : ℝ) : a^2 = 9 → a^4 = 81 := by
  sorry

end square_nine_implies_fourth_power_eightyone_l1404_140464


namespace flagpole_distance_l1404_140463

/-- Given a street of length 11.5 meters with 6 flagpoles placed at regular intervals,
    including both ends, the distance between adjacent flagpoles is 2.3 meters. -/
theorem flagpole_distance (street_length : ℝ) (num_flagpoles : ℕ) :
  street_length = 11.5 ∧ num_flagpoles = 6 →
  (street_length / (num_flagpoles - 1 : ℝ)) = 2.3 := by
  sorry

end flagpole_distance_l1404_140463


namespace notebook_cost_l1404_140429

/-- The cost of a notebook and pencil, given their relationship -/
theorem notebook_cost (notebook_cost pencil_cost : ℝ)
  (total_cost : notebook_cost + pencil_cost = 3.20)
  (cost_difference : notebook_cost = pencil_cost + 2.50) :
  notebook_cost = 2.85 := by
  sorry

end notebook_cost_l1404_140429


namespace max_profit_is_2180_l1404_140441

/-- Represents the production plan for items A and B -/
structure ProductionPlan where
  itemA : ℕ
  itemB : ℕ

/-- Calculates the profit for a given production plan -/
def profit (plan : ProductionPlan) : ℕ :=
  80 * plan.itemA + 100 * plan.itemB

/-- Checks if a production plan is feasible given the resource constraints -/
def isFeasible (plan : ProductionPlan) : Prop :=
  10 * plan.itemA + 70 * plan.itemB ≤ 700 ∧
  23 * plan.itemA + 40 * plan.itemB ≤ 642

/-- Theorem stating that the maximum profit is 2180 thousand rubles -/
theorem max_profit_is_2180 :
  ∃ (optimalPlan : ProductionPlan),
    isFeasible optimalPlan ∧
    profit optimalPlan = 2180 ∧
    ∀ (plan : ProductionPlan), isFeasible plan → profit plan ≤ 2180 := by
  sorry

end max_profit_is_2180_l1404_140441


namespace brownie_pieces_l1404_140426

theorem brownie_pieces (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 30)
  (h2 : pan_width = 24)
  (h3 : piece_length = 3)
  (h4 : piece_width = 4) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 := by
  sorry

end brownie_pieces_l1404_140426


namespace union_covers_reals_l1404_140403

open Set Real

theorem union_covers_reals (A B : Set ℝ) (a : ℝ) :
  A = Iic 0 ∧ B = Ioi a ∧ A ∪ B = univ ↔ a ≤ 0 := by
  sorry

end union_covers_reals_l1404_140403


namespace least_valid_number_l1404_140474

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ (n % 100 = n / 10)

theorem least_valid_number : 
  (∃ (n : ℕ), is_valid_number n) ∧ 
  (∀ (m : ℕ), is_valid_number m → m ≥ 900) :=
sorry

end least_valid_number_l1404_140474


namespace harmonious_example_harmonious_rational_sum_harmonious_rational_ratio_l1404_140416

/-- A pair of real numbers (a, b) is harmonious if a^2 + b and a + b^2 are both rational. -/
def Harmonious (a b : ℝ) : Prop :=
  (∃ q₁ : ℚ, a^2 + b = q₁) ∧ (∃ q₂ : ℚ, a + b^2 = q₂)

theorem harmonious_example :
  Harmonious (Real.sqrt 2 + 1/2) (1/2 - Real.sqrt 2) := by sorry

theorem harmonious_rational_sum {a b : ℝ} (h : Harmonious a b) (hs : ∃ q : ℚ, a + b = q) (hne : a + b ≠ 1) :
  ∃ (q₁ q₂ : ℚ), a = q₁ ∧ b = q₂ := by sorry

theorem harmonious_rational_ratio {a b : ℝ} (h : Harmonious a b) (hr : ∃ q : ℚ, a = q * b) :
  ∃ (q₁ q₂ : ℚ), a = q₁ ∧ b = q₂ := by sorry

end harmonious_example_harmonious_rational_sum_harmonious_rational_ratio_l1404_140416


namespace equation_substitution_l1404_140455

theorem equation_substitution (x y : ℝ) :
  (y = 2 * x - 1) → (2 * x - 3 * y = 5) → (2 * x - 6 * x + 3 = 5) :=
by
  sorry

end equation_substitution_l1404_140455


namespace prism_coloring_iff_divisible_by_three_l1404_140459

/-- A prism with an n-gon base -/
structure Prism (n : ℕ) where
  base : Fin n → Fin 3  -- coloring of the base
  top : Fin n → Fin 3   -- coloring of the top

/-- Check if a coloring is valid for a prism -/
def is_valid_coloring (n : ℕ) (p : Prism n) : Prop :=
  ∀ (i : Fin n),
    -- Each vertex is connected to all three colors
    (∃ j, p.base j ≠ p.base i ∧ p.base j ≠ p.top i) ∧
    (∃ j, p.top j ≠ p.base i ∧ p.top j ≠ p.top i) ∧
    p.base i ≠ p.top i

theorem prism_coloring_iff_divisible_by_three (n : ℕ) :
  (∃ p : Prism n, is_valid_coloring n p) ↔ 3 ∣ n :=
sorry

end prism_coloring_iff_divisible_by_three_l1404_140459


namespace shekar_marks_problem_l1404_140491

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  science : ℕ
  socialStudies : ℕ
  english : ℕ
  biology : ℕ
  mathematics : ℕ

/-- Calculates the average marks of a student -/
def averageMarks (marks : StudentMarks) : ℚ :=
  (marks.science + marks.socialStudies + marks.english + marks.biology + marks.mathematics) / 5

/-- Shekar's marks problem -/
theorem shekar_marks_problem (shekar : StudentMarks)
    (h1 : shekar.science = 65)
    (h2 : shekar.socialStudies = 82)
    (h3 : shekar.english = 67)
    (h4 : shekar.biology = 95)
    (h5 : averageMarks shekar = 77) :
    shekar.mathematics = 76 := by
  sorry

end shekar_marks_problem_l1404_140491


namespace solve_equation_l1404_140405

theorem solve_equation (x : ℝ) (h : 8 * (2 + 1 / x) = 18) : x = 4 := by
  sorry

end solve_equation_l1404_140405


namespace sequence_q_value_max_q_value_l1404_140442

-- Define the arithmetic sequence a_n
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

-- Define the geometric sequence b_n
def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

-- Define the set E
structure E where
  m : ℕ+
  p : ℕ+
  r : ℕ+
  h_order : m < p ∧ p < r

theorem sequence_q_value
  (a₁ d b₁ q : ℝ)
  (hq : q ≠ 1 ∧ q ≠ -1)
  (h_equality : arithmetic_sequence a₁ d 1 + geometric_sequence b₁ q 2 =
                arithmetic_sequence a₁ d 2 + geometric_sequence b₁ q 3 ∧
                arithmetic_sequence a₁ d 2 + geometric_sequence b₁ q 3 =
                arithmetic_sequence a₁ d 3 + geometric_sequence b₁ q 1) :
  q = -1/2 :=
sorry

theorem max_q_value
  (a₁ d b₁ q : ℝ)
  (e : E)
  (hq : q ≠ 1 ∧ q ≠ -1)
  (h_arithmetic : ∃ (k : ℝ), k > 1 ∧ e.p = e.m + k ∧ e.r = e.p + k)
  (h_equality : arithmetic_sequence a₁ d e.m + geometric_sequence b₁ q e.p =
                arithmetic_sequence a₁ d e.p + geometric_sequence b₁ q e.r ∧
                arithmetic_sequence a₁ d e.p + geometric_sequence b₁ q e.r =
                arithmetic_sequence a₁ d e.r + geometric_sequence b₁ q e.m) :
  q ≤ -(1/2)^(1/3) :=
sorry

end sequence_q_value_max_q_value_l1404_140442


namespace science_fair_participants_l1404_140400

theorem science_fair_participants (total : ℕ) (j s : ℕ) : 
  total = 240 →
  j + s = total →
  (3 * j) / 4 = s / 2 →
  (3 * j) / 4 + s / 2 = 144 :=
by sorry

end science_fair_participants_l1404_140400


namespace imaginary_part_of_i_l1404_140452

theorem imaginary_part_of_i : Complex.im i = 1 := by
  sorry

end imaginary_part_of_i_l1404_140452


namespace perpendicular_lines_parameter_l1404_140480

/-- Given two lines ax + y - 1 = 0 and 4x + (a - 5)y - 2 = 0 that are perpendicular,
    prove that a = 1 -/
theorem perpendicular_lines_parameter (a : ℝ) :
  (∃ x y, a * x + y - 1 = 0 ∧ 4 * x + (a - 5) * y - 2 = 0) →
  (∀ x₁ y₁ x₂ y₂, 
    (a * x₁ + y₁ - 1 = 0 ∧ 4 * x₁ + (a - 5) * y₁ - 2 = 0) →
    (a * x₂ + y₂ - 1 = 0 ∧ 4 * x₂ + (a - 5) * y₂ - 2 = 0) →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    (a * (x₂ - x₁) + (y₂ - y₁)) * (4 * (x₂ - x₁) + (a - 5) * (y₂ - y₁)) = 0) →
  a = 1 :=
sorry

end perpendicular_lines_parameter_l1404_140480


namespace cylinder_height_relationship_l1404_140422

/-- Represents a right circular cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Given two cylinders with equal volumes and the second radius 20% larger than the first,
    prove that the height of the first cylinder is 44% more than the height of the second --/
theorem cylinder_height_relationship (c1 c2 : Cylinder) 
    (h_volume : c1.radius^2 * c1.height = c2.radius^2 * c2.height)
    (h_radius : c2.radius = 1.2 * c1.radius) :
    c1.height = 1.44 * c2.height := by
  sorry

end cylinder_height_relationship_l1404_140422


namespace swim_team_girls_count_l1404_140493

/-- Proves that the number of girls on a swim team is 80, given the specified conditions -/
theorem swim_team_girls_count : 
  ∀ (boys girls : ℕ), 
  girls = 5 * boys → 
  boys + girls = 96 → 
  girls = 80 := by
sorry

end swim_team_girls_count_l1404_140493


namespace min_value_theorem_l1404_140425

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  ∃ (x : ℝ), x = 2 * Real.sqrt 3 - 2 ∧ ∀ (y : ℝ), 2 * a + b + c ≥ y :=
by sorry

end min_value_theorem_l1404_140425


namespace quadratic_factorization_l1404_140445

theorem quadratic_factorization (d e f : ℤ) :
  (∀ x, x^2 + 17*x + 72 = (x + d) * (x + e)) →
  (∀ x, x^2 + 7*x - 60 = (x + e) * (x - f)) →
  d + e + f = 29 := by
sorry

end quadratic_factorization_l1404_140445


namespace alexa_vacation_fraction_is_three_fourths_l1404_140434

/-- The number of days Alexa spent on vacation -/
def alexa_vacation_days : ℕ := 7 + 2

/-- The number of days it took Joey to learn swimming -/
def joey_swimming_days : ℕ := 6

/-- The number of days it took Ethan to learn fencing tricks -/
def ethan_fencing_days : ℕ := 2 * joey_swimming_days

/-- The fraction of time Alexa spent on vacation compared to Ethan's fencing learning time -/
def alexa_vacation_fraction : ℚ := alexa_vacation_days / ethan_fencing_days

theorem alexa_vacation_fraction_is_three_fourths :
  alexa_vacation_fraction = 3 / 4 := by
  sorry

end alexa_vacation_fraction_is_three_fourths_l1404_140434


namespace isosceles_right_triangle_not_regular_polygon_l1404_140488

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  side : ℝ
  side_positive : side > 0

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  side_length : ℝ
  side_positive : side_length > 0

-- Theorem: Isosceles right triangles are not regular polygons
theorem isosceles_right_triangle_not_regular_polygon :
  ∀ (t : IsoscelesRightTriangle), ¬∃ (p : RegularPolygon), true :=
sorry

end isosceles_right_triangle_not_regular_polygon_l1404_140488


namespace board_numbers_theorem_l1404_140458

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def proper_divisors (n : ℕ) : Set ℕ :=
  {d : ℕ | d ∣ n ∧ 1 < d ∧ d < n}

theorem board_numbers_theorem (n : ℕ) (hn : is_composite n) :
  (∃ m : ℕ, proper_divisors m = {d + 1 | d ∈ proper_divisors n}) ↔ n = 4 ∨ n = 8 := by
  sorry

end board_numbers_theorem_l1404_140458


namespace chord_passes_through_fixed_point_min_distance_perpendicular_chords_l1404_140494

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line l
def line_l (x y : ℝ) : Prop := y = -1

-- Define a point on the parabola
def point_on_parabola (x y : ℝ) : Prop := parabola x y

-- Define a point on line l
def point_on_line_l (x y : ℝ) : Prop := line_l x y

-- Define the chord of tangent points
def chord_of_tangent_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  point_on_parabola x₁ y₁ ∧ point_on_parabola x₂ y₂

-- Theorem 1: The chord of tangent points passes through (0, 1)
theorem chord_passes_through_fixed_point :
  ∀ x₀ y₀ x₁ y₁ x₂ y₂ : ℝ,
  point_on_line_l x₀ y₀ →
  chord_of_tangent_points x₁ y₁ x₂ y₂ →
  ∃ t : ℝ, t * x₁ + (1 - t) * x₂ = 0 ∧ t * y₁ + (1 - t) * y₂ = 1 :=
sorry

-- Theorem 2: Minimum distance between P and Q when chords are perpendicular
theorem min_distance_perpendicular_chords :
  ∃ xP yP xQ yQ : ℝ,
  point_on_line_l xP yP ∧ point_on_line_l xQ yQ ∧
  (∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
    chord_of_tangent_points x₁ y₁ x₂ y₂ ∧
    chord_of_tangent_points x₃ y₃ x₄ y₄ →
    (x₂ - x₁) * (x₄ - x₃) + (y₂ - y₁) * (y₄ - y₃) = 0 →
    (xQ - xP)^2 + (yQ - yP)^2 ≤ (x - xP)^2 + (y - yP)^2) ∧
  xP = -2 ∧ yP = -1 ∧ xQ = 2 ∧ yQ = -1 ∧
  (xQ - xP)^2 + (yQ - yP)^2 = 16 :=
sorry

end chord_passes_through_fixed_point_min_distance_perpendicular_chords_l1404_140494


namespace linear_function_not_in_third_quadrant_l1404_140461

/-- A linear function y = kx - k where k ≠ 0 and k < 0 does not pass through the third quadrant -/
theorem linear_function_not_in_third_quadrant (k : ℝ) (h1 : k ≠ 0) (h2 : k < 0) :
  ∀ x y : ℝ, y = k * x - k → ¬(x < 0 ∧ y < 0) :=
by sorry

end linear_function_not_in_third_quadrant_l1404_140461


namespace polynomial_equality_implies_sum_l1404_140497

theorem polynomial_equality_implies_sum (a b c d e f : ℝ) :
  (∀ x : ℝ, (3*x + 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  a - b + c - d + e - f = 32 := by
  sorry

end polynomial_equality_implies_sum_l1404_140497


namespace sum_of_fourth_powers_l1404_140449

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (sum_squares : a^2 + b^2 + c^2 = 3)
  (sum_cubes : a^3 + b^3 + c^3 = 6) :
  a^4 + b^4 + c^4 = 4.5 := by
  sorry

end sum_of_fourth_powers_l1404_140449


namespace is_systematic_sampling_l1404_140408

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Stratified
  | Systematic

/-- Represents the auditorium setup and sampling process -/
structure AuditoriumSampling where
  rows : Nat
  seatsPerRow : Nat
  selectedSeatNumber : Nat

/-- Determines the sampling method based on the auditorium setup and sampling process -/
def determineSamplingMethod (setup : AuditoriumSampling) : SamplingMethod := sorry

/-- Theorem stating that the given sampling process is systematic sampling -/
theorem is_systematic_sampling (setup : AuditoriumSampling) 
  (h1 : setup.rows = 40)
  (h2 : setup.seatsPerRow = 25)
  (h3 : setup.selectedSeatNumber = 18) :
  determineSamplingMethod setup = SamplingMethod.Systematic := by sorry

end is_systematic_sampling_l1404_140408


namespace no_real_solutions_for_f_iteration_l1404_140479

def f (x : ℝ) : ℝ := x^2 + 2*x

theorem no_real_solutions_for_f_iteration :
  ¬ ∃ c : ℝ, f (f (f (f c))) = -4 := by
  sorry

end no_real_solutions_for_f_iteration_l1404_140479


namespace polynomial_factor_implies_coefficients_l1404_140465

/-- Given a polynomial px^4 + qx^3 + 40x^2 - 20x + 8 with a factor of 4x^2 - 3x + 2,
    prove that p = 0 and q = -32 -/
theorem polynomial_factor_implies_coefficients
  (p q : ℚ)
  (h : ∃ (r s : ℚ), px^4 + qx^3 + 40*x^2 - 20*x + 8 = (4*x^2 - 3*x + 2) * (r*x^2 + s*x + 4)) :
  p = 0 ∧ q = -32 := by
  sorry

end polynomial_factor_implies_coefficients_l1404_140465


namespace lcm_140_225_l1404_140478

theorem lcm_140_225 : Nat.lcm 140 225 = 6300 := by
  sorry

end lcm_140_225_l1404_140478


namespace base_equation_solution_l1404_140485

/-- Convert a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Given positive integers C and D where D = C + 2, 
    and the equation 253_C - 75_D = 124_(C+D) holds, 
    prove that C + D = 26 -/
theorem base_equation_solution (C D : Nat) 
  (h1 : C > 0) 
  (h2 : D > 0) 
  (h3 : D = C + 2) 
  (h4 : toBase10 [2, 5, 3] C - toBase10 [7, 5] D = toBase10 [1, 2, 4] (C + D)) :
  C + D = 26 := by
  sorry

end base_equation_solution_l1404_140485


namespace rationalize_and_simplify_l1404_140486

theorem rationalize_and_simplify :
  ∃ (A B C : ℤ), 
    (3 + Real.sqrt 2) / (2 - Real.sqrt 5) = 
      A + B * Real.sqrt C ∧ A * B * C = -24 :=
by sorry

end rationalize_and_simplify_l1404_140486


namespace lily_shopping_ratio_l1404_140469

theorem lily_shopping_ratio (initial_balance shirt_cost final_balance : ℕ) 
  (h1 : initial_balance = 55)
  (h2 : shirt_cost = 7)
  (h3 : final_balance = 27) :
  (initial_balance - shirt_cost - final_balance) / shirt_cost = 3 := by
sorry

end lily_shopping_ratio_l1404_140469


namespace total_sales_equals_205_l1404_140411

def apple_price : ℝ := 1.50
def orange_price : ℝ := 1.00

def morning_apples : ℕ := 40
def morning_oranges : ℕ := 30
def afternoon_apples : ℕ := 50
def afternoon_oranges : ℕ := 40

def total_sales : ℝ :=
  apple_price * (morning_apples + afternoon_apples) +
  orange_price * (morning_oranges + afternoon_oranges)

theorem total_sales_equals_205 : total_sales = 205 := by
  sorry

end total_sales_equals_205_l1404_140411


namespace triangle_side_length_l1404_140499

-- Define the triangle and circle
structure Triangle :=
  (A B C O : ℝ × ℝ)
  (circumscribed : Bool)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.circumscribed ∧ 
  dist t.B t.C = 5 ∧
  dist t.A t.B = 4 ∧
  norm (3 • (t.A - t.O) - 4 • (t.B - t.O) + (t.C - t.O)) = 10

-- Theorem statement
theorem triangle_side_length (t : Triangle) :
  is_valid_triangle t → dist t.A t.C = 4 := by
  sorry

end triangle_side_length_l1404_140499


namespace large_circle_diameter_l1404_140439

/-- The diameter of a circle that encompasses six smaller tangent circles -/
theorem large_circle_diameter (r : ℝ) (offset : ℝ) : 
  r = 4 ∧ 
  offset = 1 → 
  2 * (Real.sqrt 17 + 4) = 
    2 * (Real.sqrt ((r - offset)^2 + (2*r/2)^2) + r) :=
by sorry

end large_circle_diameter_l1404_140439


namespace cubic_polynomial_roots_l1404_140419

theorem cubic_polynomial_roots (P : ℝ → ℝ) (x y z : ℝ) :
  P = (fun t ↦ t^3 - 2*t^2 - 10*t - 3) →
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ P a = 0 ∧ P b = 0 ∧ P c = 0) →
  x + y + z = 2 →
  x*y + x*z + y*z = -10 →
  x*y*z = 3 →
  let u := x^2 * y^2 * z
  let v := x^2 * z^2 * y
  let w := y^2 * z^2 * x
  let R := fun t ↦ t^3 - (u + v + w)*t^2 + (u*v + u*w + v*w)*t - u*v*w
  R = fun t ↦ t^3 + 30*t^2 + 54*t - 243 := by sorry

end cubic_polynomial_roots_l1404_140419


namespace parabola_focus_l1404_140410

/-- A parabola is defined by its equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of a parabola is a point -/
def Focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the parabola y^2 = 4x is (1, 0) -/
theorem parabola_focus :
  ∀ (p : ℝ × ℝ), p ∈ Parabola → Focus = (1, 0) :=
sorry

end parabola_focus_l1404_140410


namespace triangle_inequality_l1404_140482

theorem triangle_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  Real.sqrt (3 * (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a))) ≥ 
  Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) := by
sorry

end triangle_inequality_l1404_140482


namespace inequality_proof_l1404_140402

theorem inequality_proof (x : ℝ) :
  x ≥ Real.rpow 7 (1/3) / Real.rpow 2 (1/3) ∧
  x < Real.rpow 373 (1/3) / Real.rpow 72 (1/3) →
  Real.sqrt (2*x + 7/x^2) + Real.sqrt (2*x - 7/x^2) < 6/x :=
by sorry

end inequality_proof_l1404_140402


namespace at_least_one_chinese_book_l1404_140471

def total_books : ℕ := 12
def chinese_books : ℕ := 10
def math_books : ℕ := 2
def drawn_books : ℕ := 3

theorem at_least_one_chinese_book :
  ∀ (selection : Finset ℕ),
  selection.card = drawn_books →
  (∀ i ∈ selection, i < total_books) →
  ∃ i ∈ selection, i < chinese_books :=
by sorry

end at_least_one_chinese_book_l1404_140471


namespace deepthi_material_usage_l1404_140427

theorem deepthi_material_usage
  (material1 : ℚ)
  (material2 : ℚ)
  (leftover : ℚ)
  (h1 : material1 = 4 / 17)
  (h2 : material2 = 3 / 10)
  (h3 : leftover = 9 / 30)
  : material1 + material2 - leftover = 4 / 17 := by
  sorry

end deepthi_material_usage_l1404_140427


namespace larger_number_is_72_l1404_140451

theorem larger_number_is_72 (x y : ℝ) : 
  5 * y = 6 * x → y - x = 12 → y = 72 := by sorry

end larger_number_is_72_l1404_140451


namespace simplify_and_evaluate_l1404_140443

theorem simplify_and_evaluate : ∀ x : ℤ, 
  -1 < x → x < 3 → x ≠ 1 → x ≠ 2 →
  (3 / (x - 1) - x - 1) * ((x - 1) / (x^2 - 4*x + 4)) = (2 + x) / (2 - x) ∧
  (0 : ℤ) ∈ {y : ℤ | -1 < y ∧ y < 3 ∧ y ≠ 1 ∧ y ≠ 2} ∧
  (2 + 0) / (2 - 0) = 1 := by
sorry

end simplify_and_evaluate_l1404_140443


namespace range_of_g_l1404_140401

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - 3

-- Define the function g as a composition of f five times
def g (x : ℝ) : ℝ := f (f (f (f (f x))))

-- State the theorem
theorem range_of_g :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 →
  ∃ y : ℝ, g x = y ∧ -1023 ≤ y ∧ y ≤ 2049 :=
sorry

end range_of_g_l1404_140401


namespace vehicle_inspection_is_systematic_sampling_l1404_140415

/-- Represents a sampling method --/
structure SamplingMethod where
  name : String
  selectionCriteria : String
  isFixedInterval : Bool

/-- Defines systematic sampling --/
def systematicSampling : SamplingMethod where
  name := "Systematic Sampling"
  selectionCriteria := "Fixed periodic interval"
  isFixedInterval := true

/-- Represents the vehicle emission inspection method --/
def vehicleInspectionMethod : SamplingMethod where
  name := "Vehicle Inspection Method"
  selectionCriteria := "License plates ending in 8"
  isFixedInterval := true

/-- Theorem stating that the vehicle inspection method is systematic sampling --/
theorem vehicle_inspection_is_systematic_sampling :
  vehicleInspectionMethod = systematicSampling :=
by sorry

end vehicle_inspection_is_systematic_sampling_l1404_140415


namespace defective_item_testing_methods_l1404_140421

theorem defective_item_testing_methods :
  let genuine_items : ℕ := 6
  let defective_items : ℕ := 4
  let total_tests : ℕ := 5
  let last_test_defective : ℕ := 1
  let genuine_in_first_four : ℕ := 1
  let defective_in_first_four : ℕ := 3

  (Nat.choose defective_items last_test_defective) *
  (Nat.choose genuine_items genuine_in_first_four) *
  (Nat.choose defective_in_first_four defective_in_first_four) *
  (Nat.factorial defective_in_first_four) = 576 :=
by
  sorry

end defective_item_testing_methods_l1404_140421


namespace range_of_a_l1404_140495

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ≤ -1
def q (a x : ℝ) : Prop := a ≤ x ∧ x < a + 2

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, q a x → p x) ∧ ¬(∀ x, p x → q a x)

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, sufficient_not_necessary a ↔ a ≤ -3 :=
by sorry

end range_of_a_l1404_140495


namespace function_max_min_on_interval_l1404_140490

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem function_max_min_on_interval (m : ℝ) :
  (∀ x ∈ Set.Icc m 0, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc m 0, f x = 3) ∧
  (∀ x ∈ Set.Icc m 0, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc m 0, f x = 2) ↔
  m ∈ Set.Icc (-2) (-1) :=
by sorry

end function_max_min_on_interval_l1404_140490


namespace integer_pairs_satisfying_equation_l1404_140496

theorem integer_pairs_satisfying_equation : 
  {(x, y) : ℤ × ℤ | (y - 2) * x^2 + (y^2 - 6*y + 8) * x = y^2 - 5*y + 62} = 
  {(8, 3), (2, 9), (-7, 9), (-7, 3), (2, -6), (8, -6)} := by
  sorry

end integer_pairs_satisfying_equation_l1404_140496


namespace two_digit_number_property_l1404_140423

theorem two_digit_number_property (N : ℕ) : 
  (N ≥ 10 ∧ N ≤ 99) →
  (4 * (N / 10) + 2 * (N % 10) = N / 2) →
  (N = 32 ∨ N = 64 ∨ N = 96) :=
by sorry

end two_digit_number_property_l1404_140423


namespace janice_earnings_l1404_140489

/-- Janice's weekly earnings calculation --/
theorem janice_earnings 
  (days_per_week : ℕ) 
  (overtime_shifts : ℕ) 
  (overtime_pay : ℝ) 
  (total_earnings : ℝ) 
  (h1 : days_per_week = 5)
  (h2 : overtime_shifts = 3)
  (h3 : overtime_pay = 15)
  (h4 : total_earnings = 195) :
  ∃ (daily_earnings : ℝ), 
    daily_earnings * days_per_week + overtime_pay * overtime_shifts = total_earnings ∧ 
    daily_earnings = 30 := by
  sorry


end janice_earnings_l1404_140489


namespace APMS_is_parallelogram_l1404_140438

-- Define the points
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the quadrilateral APMS
def APMS (P Q M S A : Point2D) : Prop :=
  M.x = (P.x + Q.x) / 2 ∧
  M.y = (P.y + Q.y) / 2 ∧
  S.x = M.x ∧
  S.y ≠ M.y

-- Define what it means for a quadrilateral to be a parallelogram
def IsParallelogram (A P M S : Point2D) : Prop :=
  (P.x - A.x = M.x - S.x ∧ P.y - A.y = M.y - S.y) ∧
  (M.x - A.x = S.x - P.x ∧ M.y - A.y = S.y - P.y)

-- Theorem statement
theorem APMS_is_parallelogram 
  (P Q M S A : Point2D) 
  (h_distinct : P ≠ Q) 
  (h_APMS : APMS P Q M S A) : 
  IsParallelogram A P M S :=
sorry

end APMS_is_parallelogram_l1404_140438


namespace root_equation_solution_l1404_140418

theorem root_equation_solution (a b c : ℕ) (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (h : ∀ (N : ℝ), N ≠ 1 → (N^2 * (N^3 * N^(4/c))^(1/b))^(1/a) = N^(17/24)) :
  b = 4 := by sorry

end root_equation_solution_l1404_140418


namespace least_integer_with_12_factors_l1404_140428

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer has exactly 12 factors -/
def has_12_factors (n : ℕ+) : Prop := num_factors n = 12

theorem least_integer_with_12_factors :
  ∃ (k : ℕ+), has_12_factors k ∧ ∀ (m : ℕ+), has_12_factors m → k ≤ m :=
by
  use 108
  sorry

end least_integer_with_12_factors_l1404_140428


namespace sin_cos_sum_one_l1404_140457

theorem sin_cos_sum_one : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end sin_cos_sum_one_l1404_140457


namespace triangle_properties_l1404_140487

open Real

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating properties of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.A ≠ π / 2)
  (h2 : 3 * sin t.A * cos t.B + (1/2) * t.b * sin (2 * t.A) = 3 * sin t.C) :
  t.a = 3 ∧ 
  (t.A = 2 * π / 3 → 
    ∃ (p : ℝ), p ≤ t.a + t.b + t.c ∧ p = 3 + 2 * Real.sqrt 3) := by
  sorry


end triangle_properties_l1404_140487


namespace quadratic_root_range_l1404_140466

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x > 1 ∧ y < 1 ∧ 
   x^2 + (a^2 - 1)*x + a - 2 = 0 ∧
   y^2 + (a^2 - 1)*y + a - 2 = 0) →
  a > -2 ∧ a < 1 :=
by sorry

end quadratic_root_range_l1404_140466


namespace circles_externally_tangent_l1404_140473

theorem circles_externally_tangent (x y : ℝ) : 
  let circle1 := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6*x = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + y^2 + 8*y + 12 = 0}
  let center1 := (3, 0)
  let center2 := (0, -4)
  let radius1 := 3
  let radius2 := 2
  (∀ (p : ℝ × ℝ), p ∈ circle1 ↔ (p.1 - center1.1)^2 + (p.2 - center1.2)^2 = radius1^2) ∧
  (∀ (p : ℝ × ℝ), p ∈ circle2 ↔ (p.1 - center2.1)^2 + (p.2 - center2.2)^2 = radius2^2) ∧
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (radius1 + radius2)^2 :=
by
  sorry

end circles_externally_tangent_l1404_140473


namespace new_salary_calculation_l1404_140448

def current_salary : ℝ := 10000
def increase_percentage : ℝ := 0.02

theorem new_salary_calculation :
  current_salary * (1 + increase_percentage) = 10200 := by
  sorry

end new_salary_calculation_l1404_140448


namespace cyclic_quadrilaterals_count_l1404_140446

/-- A quadrilateral is cyclic if it can be inscribed in a circle. -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- A square is a quadrilateral with all sides equal and all angles right angles. -/
def is_square (q : Quadrilateral) : Prop := sorry

/-- A rectangle is a quadrilateral with all angles right angles. -/
def is_rectangle (q : Quadrilateral) : Prop := sorry

/-- A rhombus is a quadrilateral with all sides equal. -/
def is_rhombus (q : Quadrilateral) : Prop := sorry

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
def is_parallelogram (q : Quadrilateral) : Prop := sorry

/-- An isosceles trapezoid is a trapezoid with the non-parallel sides equal. -/
def is_isosceles_trapezoid (q : Quadrilateral) : Prop := sorry

theorem cyclic_quadrilaterals_count :
  ∃ (s r h p t : Quadrilateral),
    is_square s ∧
    is_rectangle r ∧ ¬ is_square r ∧
    is_rhombus h ∧ ¬ is_square h ∧
    is_parallelogram p ∧ ¬ is_rectangle p ∧ ¬ is_rhombus p ∧
    is_isosceles_trapezoid t ∧ ¬ is_parallelogram t ∧
    (is_cyclic s ∧ is_cyclic r ∧ is_cyclic t ∧
     ¬ is_cyclic h ∧ ¬ is_cyclic p) :=
by sorry

end cyclic_quadrilaterals_count_l1404_140446


namespace more_girls_than_boys_l1404_140477

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 30 →
  boys + girls = total_students →
  2 * girls = 3 * boys →
  girls - boys = 6 := by
sorry

end more_girls_than_boys_l1404_140477


namespace canteen_distance_l1404_140462

/-- Given a right triangle with one leg of length 400 rods and hypotenuse of length 700 rods,
    the point on the other leg that is equidistant from both endpoints of the hypotenuse
    is approximately 1711 rods from each endpoint. -/
theorem canteen_distance (a b c : ℝ) (h1 : a = 400) (h2 : c = 700) (h3 : a^2 + b^2 = c^2) :
  let x := (2 * a^2 + 2 * b^2) / (2 * b)
  ∃ ε > 0, abs (x - 1711) < ε :=
sorry

end canteen_distance_l1404_140462


namespace james_writing_time_l1404_140409

/-- James' writing scenario -/
structure WritingScenario where
  pages_per_hour : ℕ
  total_pages : ℕ
  total_weeks : ℕ

/-- Calculate the hours James writes per night -/
def hours_per_night (s : WritingScenario) : ℚ :=
  (s.total_pages : ℚ) / (s.total_weeks * 7 * s.pages_per_hour)

/-- Theorem stating that James writes for 3 hours every night -/
theorem james_writing_time (s : WritingScenario)
  (h1 : s.pages_per_hour = 5)
  (h2 : s.total_pages = 735)
  (h3 : s.total_weeks = 7) :
  hours_per_night s = 3 := by
  sorry

#eval hours_per_night ⟨5, 735, 7⟩

end james_writing_time_l1404_140409


namespace complex_number_quadrant_l1404_140440

theorem complex_number_quadrant (z : ℂ) (h : z * (2 - Complex.I) = 1) :
  0 < z.re ∧ 0 < z.im := by sorry

end complex_number_quadrant_l1404_140440


namespace parallelogram_area_equality_l1404_140432

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Given a triangle ABC, constructs parallelogram ACDE on side AC -/
def constructParallelogramOnAC (t : Triangle) : Parallelogram := sorry

/-- Given a triangle ABC, constructs parallelogram BCFG on side BC -/
def constructParallelogramOnBC (t : Triangle) : Parallelogram := sorry

/-- Given a triangle ABC and point H, constructs parallelogram ABML on side AB 
    such that AL and BM are equal and parallel to HC -/
def constructParallelogramOnAB (t : Triangle) (H : Point) : Parallelogram := sorry

/-- Main theorem statement -/
theorem parallelogram_area_equality 
  (t : Triangle) 
  (H : Point) 
  (ACDE : Parallelogram) 
  (BCFG : Parallelogram) 
  (ABML : Parallelogram) 
  (h1 : ACDE = constructParallelogramOnAC t) 
  (h2 : BCFG = constructParallelogramOnBC t) 
  (h3 : ABML = constructParallelogramOnAB t H) :
  area ABML = area ACDE + area BCFG := by sorry

end parallelogram_area_equality_l1404_140432
