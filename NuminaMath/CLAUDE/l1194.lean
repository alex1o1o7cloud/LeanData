import Mathlib

namespace coloring_book_shelves_l1194_119409

theorem coloring_book_shelves (initial_stock : ℕ) (acquired : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 2000 →
  acquired = 5000 →
  books_per_shelf = 2 →
  (initial_stock + acquired) / books_per_shelf = 3500 := by
  sorry

end coloring_book_shelves_l1194_119409


namespace repeating_decimal_to_fraction_l1194_119448

theorem repeating_decimal_to_fraction :
  ∃ (y : ℚ), y = 0.37 + (46 / 99) / 100 ∧ y = 3709 / 9900 := by
  sorry

end repeating_decimal_to_fraction_l1194_119448


namespace monomials_not_like_terms_l1194_119497

/-- Definition of a monomial -/
structure Monomial (α : Type*) [CommRing α] :=
  (coeff : α)
  (vars : List (Nat × Nat))  -- List of (variable index, exponent) pairs

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def areLikeTerms {α : Type*} [CommRing α] (m1 m2 : Monomial α) : Prop :=
  m1.vars = m2.vars

/-- Representation of the monomial -12a^2b -/
def m1 : Monomial ℚ :=
  ⟨-12, [(1, 2), (2, 1)]⟩  -- Assuming variable indices: 1 for a, 2 for b

/-- Representation of the monomial 2ab^2/3 -/
def m2 : Monomial ℚ :=
  ⟨2/3, [(1, 1), (2, 2)]⟩

theorem monomials_not_like_terms : ¬(areLikeTerms m1 m2) := by
  sorry


end monomials_not_like_terms_l1194_119497


namespace divisibility_by_35_l1194_119455

theorem divisibility_by_35 : 
  {a : ℕ | 1 ≤ a ∧ a ≤ 105 ∧ 35 ∣ (a^3 - 1)} = 
  {1, 11, 16, 36, 46, 51, 71, 81, 86} := by sorry

end divisibility_by_35_l1194_119455


namespace functional_equation_solution_l1194_119443

theorem functional_equation_solution (f : ℝ × ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x, y) + f (y, z) + f (z, x) = 0) :
  ∃ g : ℝ → ℝ, ∀ x y : ℝ, f (x, y) = g x - g y := by
  sorry

end functional_equation_solution_l1194_119443


namespace temperature_increase_proof_l1194_119408

/-- Represents the temperature increase per century -/
def temperature_increase_per_century : ℝ := 4

/-- Represents the total number of years -/
def total_years : ℕ := 1600

/-- Represents the total temperature change over the given years -/
def total_temperature_change : ℝ := 64

/-- Represents the number of years in a century -/
def years_per_century : ℕ := 100

theorem temperature_increase_proof :
  temperature_increase_per_century * (total_years / years_per_century) = total_temperature_change := by
  sorry

end temperature_increase_proof_l1194_119408


namespace christina_transfer_l1194_119417

/-- The amount Christina transferred out of her bank account -/
def amount_transferred (initial_balance final_balance : ℕ) : ℕ :=
  initial_balance - final_balance

/-- Theorem stating that Christina transferred $69 out of her bank account -/
theorem christina_transfer :
  amount_transferred 27004 26935 = 69 := by
  sorry

end christina_transfer_l1194_119417


namespace replaced_person_age_l1194_119490

/-- Represents a group of people with their ages -/
structure AgeGroup where
  size : ℕ
  average_age : ℝ

/-- Theorem stating the age of the replaced person -/
theorem replaced_person_age (group : AgeGroup) (h1 : group.size = 10) 
  (h2 : ∃ (new_average : ℝ), new_average = group.average_age - 3) 
  (h3 : ∃ (new_person_age : ℝ), new_person_age = 18) : 
  ∃ (replaced_age : ℝ), replaced_age = 48 := by
  sorry

end replaced_person_age_l1194_119490


namespace marathon_equation_l1194_119442

/-- Represents the marathon race scenario -/
theorem marathon_equation (x : ℝ) (distance : ℝ) (speed_ratio : ℝ) (head_start : ℝ) :
  distance > 0 ∧ x > 0 ∧ speed_ratio > 1 ∧ head_start > 0 →
  (distance = 5) ∧ (speed_ratio = 1.5) ∧ (head_start = 12.5 / 60) →
  distance / x = distance / (speed_ratio * x) + head_start :=
by
  sorry

end marathon_equation_l1194_119442


namespace leo_assignment_time_theorem_l1194_119470

theorem leo_assignment_time_theorem :
  ∀ (first_part second_part third_part first_break second_break total_time : ℕ),
    first_part = 25 →
    second_part = 2 * first_part →
    first_break = 10 →
    second_break = 15 →
    total_time = 150 →
    total_time = first_part + second_part + third_part + first_break + second_break →
    third_part = 50 := by
  sorry

end leo_assignment_time_theorem_l1194_119470


namespace maximum_discount_rate_proof_l1194_119413

/-- Represents the maximum discount rate that can be applied to a product. -/
def max_discount_rate : ℝ := 8.8

/-- The cost price of the product in yuan. -/
def cost_price : ℝ := 4

/-- The original selling price of the product in yuan. -/
def original_selling_price : ℝ := 5

/-- The minimum required profit margin as a percentage. -/
def min_profit_margin : ℝ := 10

theorem maximum_discount_rate_proof :
  let discounted_price := original_selling_price * (1 - max_discount_rate / 100)
  let profit := discounted_price - cost_price
  let profit_margin := (profit / cost_price) * 100
  (profit_margin ≥ min_profit_margin) ∧
  (∀ x : ℝ, x > max_discount_rate →
    let new_discounted_price := original_selling_price * (1 - x / 100)
    let new_profit := new_discounted_price - cost_price
    let new_profit_margin := (new_profit / cost_price) * 100
    new_profit_margin < min_profit_margin) :=
by sorry

#check maximum_discount_rate_proof

end maximum_discount_rate_proof_l1194_119413


namespace raja_medicine_percentage_l1194_119482

/-- Raja's monthly expenses and savings --/
def monthly_expenses (income medicine_percentage : ℝ) : Prop :=
  let household_percentage : ℝ := 0.35
  let clothes_percentage : ℝ := 0.20
  let savings : ℝ := 15000
  household_percentage * income + 
  clothes_percentage * income + 
  medicine_percentage * income + 
  savings = income

theorem raja_medicine_percentage : 
  ∃ (medicine_percentage : ℝ), 
    monthly_expenses 37500 medicine_percentage ∧ 
    medicine_percentage = 0.05 := by
  sorry

end raja_medicine_percentage_l1194_119482


namespace f_period_and_range_l1194_119457

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3 * (Real.sin x)^2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem f_period_and_range :
  (∃ T > 0, is_periodic f T ∧ ∀ T' > 0, is_periodic f T' → T ≤ T') ∧
  (∀ y ∈ Set.range (f ∘ (fun x => x * π / 3)), -Real.sqrt 3 ≤ y ∧ y ≤ 2) :=
by sorry

end f_period_and_range_l1194_119457


namespace withdrawal_recorded_as_negative_l1194_119414

-- Define the banking system
structure BankAccount where
  balance : ℤ

-- Define deposit and withdrawal operations
def deposit (account : BankAccount) (amount : ℕ) : BankAccount :=
  { balance := account.balance + amount }

def withdraw (account : BankAccount) (amount : ℕ) : BankAccount :=
  { balance := account.balance - amount }

-- Theorem statement
theorem withdrawal_recorded_as_negative (initial_balance : ℕ) (withdrawal_amount : ℕ) :
  (withdraw (BankAccount.mk initial_balance) withdrawal_amount).balance =
  initial_balance - withdrawal_amount :=
by sorry

end withdrawal_recorded_as_negative_l1194_119414


namespace carla_book_count_l1194_119472

theorem carla_book_count (ceiling_tiles : ℕ) (tuesday_count : ℕ) : 
  ceiling_tiles = 38 → 
  tuesday_count = 301 → 
  ∃ (books : ℕ), 2 * ceiling_tiles + 3 * books = tuesday_count ∧ books = 75 :=
by sorry

end carla_book_count_l1194_119472


namespace expression_factorization_l1194_119475

theorem expression_factorization (y : ℝ) : 
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 - 9) = 6 * y^4 * (2 * y^2 + 7) := by
  sorry

end expression_factorization_l1194_119475


namespace craig_remaining_apples_l1194_119496

/-- Theorem: Craig's remaining apples after sharing -/
theorem craig_remaining_apples (initial_apples shared_apples : ℕ) 
  (h1 : initial_apples = 20)
  (h2 : shared_apples = 7) :
  initial_apples - shared_apples = 13 := by
  sorry

end craig_remaining_apples_l1194_119496


namespace boat_length_l1194_119405

/-- The length of a boat given specific conditions -/
theorem boat_length (breadth : Real) (sinking_depth : Real) (man_mass : Real) (water_density : Real) :
  breadth = 3 ∧ 
  sinking_depth = 0.01 ∧ 
  man_mass = 210 ∧ 
  water_density = 1000 →
  ∃ (length : Real), length = 7 ∧ 
    man_mass = water_density * (length * breadth * sinking_depth) :=
by sorry

end boat_length_l1194_119405


namespace x_to_y_value_l1194_119461

theorem x_to_y_value (x y : ℝ) (h : (x - 2)^2 + Real.sqrt (y + 1) = 0) : x^y = 1/2 := by
  sorry

end x_to_y_value_l1194_119461


namespace geometric_sequence_common_ratio_l1194_119471

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 2 →                    -- a_1 = 2
  (a 1 + a 2 + a 3 = 26) →     -- S_3 = 26
  q = 3 ∨ q = -4 :=            -- conclusion: q is 3 or -4
by sorry

end geometric_sequence_common_ratio_l1194_119471


namespace organizing_teams_count_l1194_119473

theorem organizing_teams_count (total_members senior_members team_size : ℕ) 
  (h1 : total_members = 12)
  (h2 : senior_members = 5)
  (h3 : team_size = 5) :
  (Nat.choose total_members team_size) - 
  ((Nat.choose (total_members - senior_members) team_size) + 
   (Nat.choose senior_members 1 * Nat.choose (total_members - senior_members) (team_size - 1))) = 596 := by
sorry

end organizing_teams_count_l1194_119473


namespace division_remainder_l1194_119431

theorem division_remainder (N : ℕ) : N = 7 * 5 + 0 → N % 11 = 2 := by
  sorry

end division_remainder_l1194_119431


namespace correct_subtraction_l1194_119419

theorem correct_subtraction (x : ℤ) (h : x - 63 = 8) : x - 36 = 35 := by
  sorry

end correct_subtraction_l1194_119419


namespace calculation_one_l1194_119491

theorem calculation_one : (-3/8) + (-5/8) * (-6) = 27/8 := by sorry

end calculation_one_l1194_119491


namespace john_travel_solution_l1194_119450

/-- Represents the problem of calculating the distance John travels -/
def john_travel_problem (initial_speed : ℝ) (speed_increase : ℝ) (initial_time : ℝ) 
  (late_time : ℝ) (early_time : ℝ) : Prop :=
  ∃ (total_distance : ℝ) (total_time : ℝ),
    initial_speed * initial_time = initial_speed ∧
    total_distance = initial_speed * (total_time + late_time / 60) ∧
    total_distance = initial_speed * initial_time + 
      (initial_speed + speed_increase) * (total_time - initial_time - early_time / 60) ∧
    total_distance = 123.4375

/-- The theorem stating that the solution to John's travel problem exists -/
theorem john_travel_solution : 
  john_travel_problem 25 20 1 1.5 0.25 := by sorry

end john_travel_solution_l1194_119450


namespace line_plane_relationship_l1194_119445

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (m : Line) (α β : Plane) 
  (h1 : perpPlanes α β) 
  (h2 : perp m α) : 
  para m β ∨ subset m β := by
  sorry

end line_plane_relationship_l1194_119445


namespace age_difference_l1194_119411

theorem age_difference (man_age son_age : ℕ) : 
  son_age = 24 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 26 := by
sorry

end age_difference_l1194_119411


namespace coefficient_equals_20th_term_l1194_119494

theorem coefficient_equals_20th_term : 
  let binomial (n k : ℕ) := Nat.choose n k
  let coefficient := binomial 5 4 + binomial 6 4 + binomial 7 4
  let a (n : ℕ) := 3 * n - 5
  coefficient = a 20 := by sorry

end coefficient_equals_20th_term_l1194_119494


namespace max_y_value_l1194_119449

theorem max_y_value (x y : ℝ) (h1 : x > 0) (h2 : x * y * (x + y) = x - y) : 
  y ≤ 1/3 ∧ ∃ (y0 : ℝ), y0 * x * (x + y0) = x - y0 ∧ y0 = 1/3 :=
sorry

end max_y_value_l1194_119449


namespace picture_processing_time_l1194_119433

theorem picture_processing_time (num_pictures : ℕ) (processing_time_per_picture : ℕ) : 
  num_pictures = 960 → 
  processing_time_per_picture = 2 → 
  (num_pictures * processing_time_per_picture) / 60 = 32 := by
sorry

end picture_processing_time_l1194_119433


namespace congruence_solution_l1194_119441

theorem congruence_solution (p q : Nat) (n : Nat) : 
  Nat.Prime p → Nat.Prime q → Odd p → Odd q → n > 1 →
  (q ^ (n + 2) % (p ^ n) = 3 ^ (n + 2) % (p ^ n)) →
  (p ^ (n + 2) % (q ^ n) = 3 ^ (n + 2) % (q ^ n)) →
  (p = 3 ∧ q = 3) := by
  sorry

end congruence_solution_l1194_119441


namespace sandwich_change_calculation_l1194_119435

theorem sandwich_change_calculation (num_sandwiches : ℕ) (cost_per_sandwich : ℕ) (amount_paid : ℕ) : 
  num_sandwiches = 3 → cost_per_sandwich = 5 → amount_paid = 20 → 
  amount_paid - (num_sandwiches * cost_per_sandwich) = 5 := by
  sorry

end sandwich_change_calculation_l1194_119435


namespace constant_point_on_line_l1194_119493

/-- The line equation passing through a constant point regardless of m -/
def line_equation (m x y : ℝ) : Prop :=
  (m + 2) * x - (2 * m - 1) * y = 3 * m - 4

/-- The theorem stating that (-1, -2) satisfies the line equation for all m -/
theorem constant_point_on_line :
  ∀ m : ℝ, line_equation m (-1) (-2) :=
by sorry

end constant_point_on_line_l1194_119493


namespace f_2011_equals_2011_l1194_119495

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the main property of f
variable (h : ∀ a b : ℝ, f (a * f b) = a * b)

-- Theorem statement
theorem f_2011_equals_2011 : f 2011 = 2011 := by
  sorry

end f_2011_equals_2011_l1194_119495


namespace floor_sqrt_12_squared_l1194_119421

theorem floor_sqrt_12_squared : ⌊Real.sqrt 12⌋^2 = 9 := by
  sorry

end floor_sqrt_12_squared_l1194_119421


namespace largest_common_value_l1194_119451

/-- The first arithmetic progression -/
def seq1 (n : ℕ) : ℕ := 4 + 5 * n

/-- The second arithmetic progression -/
def seq2 (m : ℕ) : ℕ := 5 + 10 * m

/-- A common term of both sequences -/
def common_term (k : ℕ) : ℕ := 4 + 10 * k

theorem largest_common_value :
  (∃ n m : ℕ, seq1 n = seq2 m ∧ seq1 n < 1000) ∧
  (∀ n m : ℕ, seq1 n = seq2 m → seq1 n < 1000 → seq1 n ≤ 994) ∧
  (∃ k : ℕ, common_term k = 994 ∧ common_term k = seq1 (2 * k) ∧ common_term k = seq2 k) :=
sorry

end largest_common_value_l1194_119451


namespace parabola_equation_l1194_119478

/-- A parabola with vertex at the origin and axis at x = 3/2 has the equation y² = -6x -/
theorem parabola_equation (y x : ℝ) : 
  (∀ (p : ℝ), p > 0 → y^2 = -2*p*x) → -- General equation of parabola with vertex at origin
  (3/2 : ℝ) = p/2 →                   -- Axis of parabola is at x = 3/2
  y^2 = -6*x :=                       -- Equation to be proved
by sorry

end parabola_equation_l1194_119478


namespace fraction_of_25_l1194_119465

theorem fraction_of_25 : 
  ∃ (x : ℚ), x * 25 + 8 = 70 * 40 / 100 ∧ x = 4 / 5 := by
  sorry

end fraction_of_25_l1194_119465


namespace meal_cost_calculation_l1194_119492

theorem meal_cost_calculation (initial_friends : ℕ) (additional_friends : ℕ) 
  (cost_decrease : ℚ) (total_cost : ℚ) : 
  initial_friends = 4 →
  additional_friends = 5 →
  cost_decrease = 6 →
  (total_cost / initial_friends.cast) - (total_cost / (initial_friends + additional_friends).cast) = cost_decrease →
  total_cost = 216/5 := by
  sorry

end meal_cost_calculation_l1194_119492


namespace boat_upstream_speed_l1194_119402

/-- The speed of a boat upstream given its speed in still water and the speed of the current. -/
def speed_upstream (speed_still : ℝ) (speed_current : ℝ) : ℝ :=
  speed_still - speed_current

/-- Theorem: Given a boat with speed 50 km/h in still water and a current with speed 20 km/h,
    the speed of the boat upstream is 30 km/h. -/
theorem boat_upstream_speed :
  let speed_still : ℝ := 50
  let speed_current : ℝ := 20
  speed_upstream speed_still speed_current = 30 := by
  sorry

end boat_upstream_speed_l1194_119402


namespace problem_solution_l1194_119426

theorem problem_solution (x y : ℚ) 
  (h1 : x + y = 2/3)
  (h2 : x/y = 2/3) : 
  x - y = -2/15 := by sorry

end problem_solution_l1194_119426


namespace geometric_sequence_sum_l1194_119498

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 + a 2 + a 3 = 1 →
  a 2 + a 3 + a 4 = 2 →
  a 6 + a 7 + a 8 = 32 := by
  sorry

end geometric_sequence_sum_l1194_119498


namespace book_pages_l1194_119412

theorem book_pages (x : ℕ) (h1 : x > 0) (h2 : x + (x + 1) = 137) : x + 1 = 69 := by
  sorry

end book_pages_l1194_119412


namespace f_is_quadratic_l1194_119466

/-- A quadratic equation in terms of x is of the form ax^2 + bx + c = 0, where a ≠ 0, b, and c are real numbers. -/
def IsQuadraticEquation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 2x^2 - 1 = 0 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : IsQuadraticEquation f := by
  sorry

end f_is_quadratic_l1194_119466


namespace smallest_bob_number_l1194_119469

def alice_number : ℕ := 36

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ m → p ∣ n)

def is_multiple_of_five (n : ℕ) : Prop :=
  5 ∣ n

theorem smallest_bob_number :
  ∃ (bob_number : ℕ),
    has_all_prime_factors bob_number alice_number ∧
    is_multiple_of_five bob_number ∧
    (∀ k : ℕ, k < bob_number →
      ¬(has_all_prime_factors k alice_number ∧ is_multiple_of_five k)) ∧
    bob_number = 30 :=
by sorry

end smallest_bob_number_l1194_119469


namespace factor_expression_l1194_119422

theorem factor_expression (x : ℝ) : 12 * x^2 - 6 * x = 6 * x * (2 * x - 1) := by
  sorry

end factor_expression_l1194_119422


namespace collinear_implies_coplanar_exist_coplanar_non_collinear_l1194_119437

-- Define a Point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a predicate for three points being collinear
def collinear (p q r : Point3D) : Prop := sorry

-- Define a predicate for four points being coplanar
def coplanar (p q r s : Point3D) : Prop := sorry

-- Theorem: If three out of four points are collinear, then all four points are coplanar
theorem collinear_implies_coplanar (p q r s : Point3D) :
  (collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) →
  coplanar p q r s :=
sorry

-- Theorem: There exist four coplanar points where no three are collinear
theorem exist_coplanar_non_collinear :
  ∃ (p q r s : Point3D), coplanar p q r s ∧
    ¬(collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) :=
sorry

end collinear_implies_coplanar_exist_coplanar_non_collinear_l1194_119437


namespace polynomial_coefficient_B_l1194_119446

-- Define the polynomial
def polynomial (z A B C D : ℤ) : ℤ := z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 20

-- Define the property that all roots are positive integers
def all_roots_positive_integers (p : ℤ → ℤ) : Prop :=
  ∀ r : ℤ, p r = 0 → r > 0

-- State the theorem
theorem polynomial_coefficient_B :
  ∀ A B C D : ℤ,
  all_roots_positive_integers (polynomial · A B C D) →
  B = -160 := by
sorry

end polynomial_coefficient_B_l1194_119446


namespace plane_equation_proof_l1194_119415

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by its direction ratios -/
structure Line3D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A plane in 3D space defined by its equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

/-- Check if a point lies on a plane -/
def point_on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if a line is contained in a plane -/
def line_in_plane (l : Line3D) (plane : Plane) : Prop :=
  plane.A * l.a + plane.B * l.b + plane.C * l.c = 0

/-- The main theorem -/
theorem plane_equation_proof (p : Point3D) (l : Line3D) :
  p = Point3D.mk 0 7 (-7) →
  l = Line3D.mk (-3) 2 1 →
  let plane := Plane.mk 1 1 1 0
  point_on_plane p plane ∧ line_in_plane l plane := by sorry

end plane_equation_proof_l1194_119415


namespace intersection_theorem_l1194_119424

def setA : Set ℝ := {x | (x + 3) * (x - 1) ≤ 0}

def setB : Set ℝ := {x | ∃ y, y = Real.log (x^2 - x - 2)}

theorem intersection_theorem : 
  setA ∩ (setB.compl) = {x | -1 ≤ x ∧ x ≤ 1} := by sorry

end intersection_theorem_l1194_119424


namespace technician_salary_l1194_119483

/-- Proves that the average salary of technicians is 12000 given the workshop conditions --/
theorem technician_salary (total_workers : ℕ) (technicians : ℕ) (avg_salary : ℕ) (non_tech_salary : ℕ) :
  total_workers = 21 →
  technicians = 7 →
  avg_salary = 8000 →
  non_tech_salary = 6000 →
  (avg_salary * total_workers = 12000 * technicians + non_tech_salary * (total_workers - technicians)) :=
by
  sorry

#check technician_salary

end technician_salary_l1194_119483


namespace range_of_a_l1194_119454

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 5, log10 (x^2 + a*x) = 1) → 
  a ∈ Set.Icc (-3) 9 :=
by sorry

end range_of_a_l1194_119454


namespace second_day_student_tickets_second_day_student_tickets_is_ten_l1194_119456

/-- The price of a student ticket -/
def student_ticket_price : ℕ := 9

/-- The total revenue from the first day of sales -/
def first_day_revenue : ℕ := 79

/-- The total revenue from the second day of sales -/
def second_day_revenue : ℕ := 246

/-- The number of senior citizen tickets sold on the first day -/
def first_day_senior_tickets : ℕ := 4

/-- The number of student tickets sold on the first day -/
def first_day_student_tickets : ℕ := 3

/-- The number of senior citizen tickets sold on the second day -/
def second_day_senior_tickets : ℕ := 12

/-- Calculates the price of a senior citizen ticket based on the first day's sales -/
def senior_ticket_price : ℕ := 
  (first_day_revenue - student_ticket_price * first_day_student_tickets) / first_day_senior_tickets

theorem second_day_student_tickets : ℕ := 
  (second_day_revenue - senior_ticket_price * second_day_senior_tickets) / student_ticket_price

theorem second_day_student_tickets_is_ten : second_day_student_tickets = 10 := by
  sorry

end second_day_student_tickets_second_day_student_tickets_is_ten_l1194_119456


namespace binomial_probability_two_l1194_119499

/-- A random variable following a binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ  -- The random variable

/-- The probability mass function for a binomial distribution -/
def binomialProbability (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

/-- The theorem stating that P(X=2) = 80/243 for X ~ B(6, 1/3) -/
theorem binomial_probability_two (X : BinomialDistribution 6 (1/3)) :
  binomialProbability 6 (1/3) 2 = 80/243 := by
  sorry

end binomial_probability_two_l1194_119499


namespace f_2017_value_l1194_119479

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2017_value (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_value : f (-1) = 6) :
  f 2017 = -6 := by
sorry

end f_2017_value_l1194_119479


namespace tan_cos_tan_equality_l1194_119439

theorem tan_cos_tan_equality : Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end tan_cos_tan_equality_l1194_119439


namespace real_part_of_complex_number_l1194_119476

theorem real_part_of_complex_number (i : ℂ) (h : i^2 = -1) : 
  Complex.re ((-1 + 2*i)*i) = -2 := by
  sorry

end real_part_of_complex_number_l1194_119476


namespace quadratic_inequality_implies_equality_l1194_119485

theorem quadratic_inequality_implies_equality (x : ℝ) :
  -2 * x^2 + 5 * x - 2 > 0 →
  Real.sqrt (4 * x^2 - 4 * x + 1) + 2 * abs (x - 2) = 3 := by
sorry

end quadratic_inequality_implies_equality_l1194_119485


namespace six_digit_number_problem_l1194_119425

theorem six_digit_number_problem : ∃! n : ℕ, 
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ k : ℕ, n = 200000 + k ∧ k < 100000 ∧
  10 * k + 2 = 3 * n ∧
  n = 285714 := by
sorry

end six_digit_number_problem_l1194_119425


namespace sin_cos_45_sum_l1194_119464

theorem sin_cos_45_sum : Real.sin (π / 4) + Real.cos (π / 4) = Real.sqrt 2 := by
  sorry

end sin_cos_45_sum_l1194_119464


namespace expand_and_simplify_l1194_119444

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end expand_and_simplify_l1194_119444


namespace proposition_implication_l1194_119403

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 5) : 
  ¬ P 4 := by
  sorry

end proposition_implication_l1194_119403


namespace trigonometric_equality_iff_sum_pi_half_l1194_119440

open Real

theorem trigonometric_equality_iff_sum_pi_half 
  (α β : ℝ) (k : ℕ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : k > 0) :
  (sin α)^(k+2) / (cos β)^k + (cos α)^(k+2) / (sin β)^k = 1 ↔ α + β = π/2 :=
by sorry

end trigonometric_equality_iff_sum_pi_half_l1194_119440


namespace trebled_result_l1194_119458

theorem trebled_result (initial_number : ℕ) : 
  initial_number = 17 → 
  3 * (2 * initial_number + 5) = 117 := by
  sorry

end trebled_result_l1194_119458


namespace fraction_sum_equality_l1194_119400

theorem fraction_sum_equality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) = 1 := by
  sorry

end fraction_sum_equality_l1194_119400


namespace odd_perfect_number_l1194_119460

/-- Sum of positive divisors of n -/
def sigma (n : ℕ) : ℕ := sorry

/-- A number is perfect if σ(n) = 2n -/
def isPerfect (n : ℕ) : Prop := sigma n = 2 * n

theorem odd_perfect_number (n : ℕ) (h : n > 0) (h_sigma : (sigma n : ℚ) / n = 5 / 3) :
  isPerfect (5 * n) ∧ Odd (5 * n) := by sorry

end odd_perfect_number_l1194_119460


namespace marie_lost_erasers_l1194_119459

/-- The number of erasers Marie lost -/
def erasers_lost (initial final : ℕ) : ℕ := initial - final

/-- Theorem stating that Marie lost 42 erasers -/
theorem marie_lost_erasers : 
  let initial := 95
  let final := 53
  erasers_lost initial final = 42 := by
sorry

end marie_lost_erasers_l1194_119459


namespace quadratic_equation_solution_l1194_119416

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, x^2 + 2*x = 0 ↔ x = x₁ ∨ x = x₂) ∧ x₁ = 0 ∧ x₂ = -2 := by
  sorry

end quadratic_equation_solution_l1194_119416


namespace semicircle_perimeter_approx_l1194_119463

/-- The perimeter of a semicircle with radius 7 is approximately 35.99 -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 7
  let perimeter : ℝ := 2 * r + π * r
  ∃ ε > 0, abs (perimeter - 35.99) < ε :=
by sorry

end semicircle_perimeter_approx_l1194_119463


namespace money_left_after_purchase_l1194_119477

theorem money_left_after_purchase (initial_amount spent_amount : ℕ) : 
  initial_amount = 90 → spent_amount = 78 → initial_amount - spent_amount = 12 := by
  sorry

end money_left_after_purchase_l1194_119477


namespace y_value_proof_l1194_119452

theorem y_value_proof (x y : ℕ+) 
  (h1 : y = (x : ℚ) * (1/4 : ℚ) * (1/2 : ℚ))
  (h2 : (y : ℚ) * (x : ℚ) / 100 = 100) :
  y = 35 := by
  sorry

end y_value_proof_l1194_119452


namespace fifth_power_equality_l1194_119453

theorem fifth_power_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := by
  sorry

end fifth_power_equality_l1194_119453


namespace cost_of_flour_l1194_119474

/-- Given the total cost of flour and cake stand, and the cost of the cake stand,
    prove that the cost of flour is $5. -/
theorem cost_of_flour (total_cost cake_stand_cost : ℕ)
  (h1 : total_cost = 33)
  (h2 : cake_stand_cost = 28) :
  total_cost - cake_stand_cost = 5 := by
  sorry

end cost_of_flour_l1194_119474


namespace gas_station_lighter_price_l1194_119430

/-- The cost of a single lighter at the gas station -/
def gas_station_price : ℝ := 1.75

/-- The cost of a pack of 12 lighters on Amazon -/
def amazon_pack_price : ℝ := 5

/-- The number of lighters in a pack on Amazon -/
def lighters_per_pack : ℕ := 12

/-- The number of lighters Amanda is considering buying -/
def total_lighters : ℕ := 24

/-- The amount saved by buying online instead of at the gas station -/
def savings : ℝ := 32

theorem gas_station_lighter_price :
  gas_station_price = 1.75 ∧
  amazon_pack_price * (total_lighters / lighters_per_pack) + savings =
    gas_station_price * total_lighters :=
by sorry

end gas_station_lighter_price_l1194_119430


namespace solution_satisfies_equation_l1194_119434

variable (x : ℝ) (y : ℝ → ℝ) (C : ℝ)

noncomputable def solution (x : ℝ) (y : ℝ → ℝ) (C : ℝ) : Prop :=
  x * y x - 1 / (x * y x) - 2 * Real.log (abs (y x)) = C

def differential_equation (x : ℝ) (y : ℝ → ℝ) : Prop :=
  (1 + x^2 * (y x)^2) * y x + (x * y x - 1)^2 * x * (deriv y x) = 0

theorem solution_satisfies_equation :
  solution x y C → differential_equation x y :=
sorry

end solution_satisfies_equation_l1194_119434


namespace subset_intersection_problem_l1194_119410

theorem subset_intersection_problem (a : ℝ) :
  let A := { x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5 }
  let B := { x : ℝ | 3 ≤ x ∧ x ≤ 22 }
  (A ⊆ A ∩ B) ↔ (6 ≤ a ∧ a ≤ 9) :=
by sorry

end subset_intersection_problem_l1194_119410


namespace middle_term_of_arithmetic_sequence_l1194_119418

-- Define an arithmetic sequence
def is_arithmetic_sequence (a b c : ℤ) : Prop :=
  b - a = c - b

-- State the theorem
theorem middle_term_of_arithmetic_sequence :
  ∀ y : ℤ, is_arithmetic_sequence (3^2) y (3^4) → y = 45 :=
by
  sorry

end middle_term_of_arithmetic_sequence_l1194_119418


namespace necessary_but_not_sufficient_condition_l1194_119481

theorem necessary_but_not_sufficient_condition (x : ℝ) : 
  (∀ y : ℝ, 2*y > 2 → y > -1) ∧ 
  (∃ z : ℝ, z > -1 ∧ ¬(2*z > 2)) := by
  sorry

end necessary_but_not_sufficient_condition_l1194_119481


namespace circle_radius_from_area_l1194_119488

theorem circle_radius_from_area (A : Real) (r : Real) :
  A = Real.pi * r^2 → A = 64 * Real.pi → r = 8 := by
  sorry

end circle_radius_from_area_l1194_119488


namespace amelia_position_100_l1194_119447

/-- Represents a position on the coordinate plane -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a direction -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Defines Amelia's movement pattern -/
def ameliaMove (n : Nat) : Position :=
  sorry

/-- Theorem stating Amelia's position at p₁₀₀ -/
theorem amelia_position_100 : ameliaMove 100 = Position.mk 0 19 := by
  sorry

end amelia_position_100_l1194_119447


namespace solution_quadratic_equation_l1194_119428

theorem solution_quadratic_equation :
  ∀ x : ℝ, (x - 2)^2 = 3*(x - 2) ↔ x = 2 ∨ x = 5 := by sorry

end solution_quadratic_equation_l1194_119428


namespace unit_vector_AB_l1194_119487

/-- Given points A(1,3) and B(4,-1), the unit vector in the same direction as vector AB is (3/5, -4/5) -/
theorem unit_vector_AB (A B : ℝ × ℝ) (h : A = (1, 3) ∧ B = (4, -1)) :
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let magnitude : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  (AB.1 / magnitude, AB.2 / magnitude) = (3/5, -4/5) := by
  sorry


end unit_vector_AB_l1194_119487


namespace smarties_leftover_l1194_119467

theorem smarties_leftover (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 := by
  sorry

end smarties_leftover_l1194_119467


namespace cosine_angle_special_vectors_l1194_119486

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem cosine_angle_special_vectors (a b : E) (ha : a ≠ 0) (hb : b ≠ 0)
  (norm_a : ‖a‖ = 1) (norm_b : ‖b‖ = 1) (norm_sum : ‖a + 2 • b‖ = 1) :
  inner a b / (‖a‖ * ‖b‖) = -1 :=
sorry

end cosine_angle_special_vectors_l1194_119486


namespace area_of_triangle_BDE_l1194_119420

noncomputable section

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Angle between three points in 3D space -/
def angle (p q r : Point3D) : ℝ := sorry

/-- Check if two lines are parallel in 3D space -/
def parallel_lines (p1 q1 p2 q2 : Point3D) : Prop := sorry

/-- Check if a plane is parallel to a line in 3D space -/
def plane_parallel_to_line (p1 p2 p3 l1 l2 : Point3D) : Prop := sorry

/-- Calculate the area of a triangle given its three vertices -/
def triangle_area (p q r : Point3D) : ℝ := sorry

theorem area_of_triangle_BDE (A B C D E : Point3D)
  (h1 : distance A B = 3)
  (h2 : distance B C = 3)
  (h3 : distance C D = 3)
  (h4 : distance D E = 3)
  (h5 : distance E A = 3)
  (h6 : angle A B C = Real.pi / 2)
  (h7 : angle C D E = Real.pi / 2)
  (h8 : angle D E A = Real.pi / 2)
  (h9 : plane_parallel_to_line A C D B E) :
  triangle_area B D E = 9 := by
  sorry

end area_of_triangle_BDE_l1194_119420


namespace problem_statement_l1194_119427

theorem problem_statement (a b c d e : ℝ) 
  (h1 : (a + c) * (a + d) = e)
  (h2 : (b + c) * (b + d) = e)
  (h3 : e ≠ 0)
  (h4 : a ≠ b) :
  (a + c) * (b + c) - (a + d) * (b + d) = 0 := by
sorry

end problem_statement_l1194_119427


namespace frankies_pets_l1194_119401

theorem frankies_pets (cats snakes parrots dogs : ℕ) : 
  snakes = cats + 6 →
  parrots = cats - 1 →
  cats + dogs = 6 →
  cats + snakes + parrots + dogs = 19 :=
by
  sorry

end frankies_pets_l1194_119401


namespace unique_solution_implies_k_zero_l1194_119423

theorem unique_solution_implies_k_zero (a b k : ℤ) : 
  (∃! p : ℝ × ℝ, 
    (p.1 = a ∧ p.2 = b) ∧ 
    Real.sqrt (↑a - 1) + Real.sqrt (↑b - 1) = Real.sqrt (↑(a * b + k))) → 
  k = 0 :=
by sorry

end unique_solution_implies_k_zero_l1194_119423


namespace false_premise_implications_l1194_119406

theorem false_premise_implications :
  ∃ (p : Prop) (q r : Prop), 
    (¬p) ∧ (p → q) ∧ (p → r) ∧ q ∧ (¬r) := by
  -- Let p be the false premise 5 = -5
  let p := (5 = -5)
  -- Let q be the true conclusion 25 = 25
  let q := (25 = 25)
  -- Let r be the false conclusion 125 = -125
  let r := (125 = -125)
  
  have h1 : ¬p := by sorry
  have h2 : p → q := by sorry
  have h3 : p → r := by sorry
  have h4 : q := by sorry
  have h5 : ¬r := by sorry

  exact ⟨p, q, r, h1, h2, h3, h4, h5⟩

#check false_premise_implications

end false_premise_implications_l1194_119406


namespace smallest_two_digit_multiple_of_17_l1194_119489

theorem smallest_two_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 17 ∧ 
  (∀ m : ℕ, m % 17 = 0 ∧ 10 ≤ m ∧ m < 100 → n ≤ m) := by
  sorry

end smallest_two_digit_multiple_of_17_l1194_119489


namespace ratio_problem_l1194_119436

theorem ratio_problem (a b : ℝ) (h : (9*a - 4*b) / (12*a - 3*b) = 4/7) : 
  a / b = 16 / 15 := by
sorry

end ratio_problem_l1194_119436


namespace indeterminate_teachers_per_department_l1194_119404

/-- Represents a school with departments and teachers -/
structure School where
  departments : ℕ
  total_teachers : ℕ

/-- Defines a function to check if it's possible to determine exact number of teachers per department -/
def can_determine_teachers_per_department (s : School) : Prop :=
  ∃ (teachers_per_dept : ℕ), s.total_teachers = s.departments * teachers_per_dept

/-- Theorem stating that for a school with 7 departments and 140 teachers, 
    it's not always possible to determine the exact number of teachers in each department -/
theorem indeterminate_teachers_per_department :
  ¬ ∀ (s : School), s.departments = 7 ∧ s.total_teachers = 140 → can_determine_teachers_per_department s :=
by
  sorry


end indeterminate_teachers_per_department_l1194_119404


namespace new_person_weight_l1194_119468

/-- The weight of the new person given the conditions of the problem -/
theorem new_person_weight (n : ℕ) (initial_weight : ℝ) (weight_increase : ℝ) : 
  n = 9 → 
  initial_weight = 86 → 
  weight_increase = 5.5 → 
  (n : ℝ) * weight_increase + initial_weight = 135.5 := by
  sorry

end new_person_weight_l1194_119468


namespace largest_initial_number_l1194_119429

theorem largest_initial_number :
  ∃ (a b c d e : ℕ),
    189 ∉ {x : ℕ | a ∣ x ∨ b ∣ x ∨ c ∣ x ∨ d ∣ x ∨ e ∣ x} ∧
    189 + a + b + c + d + e = 200 ∧
    ∀ (n : ℕ), n > 189 →
      ¬∃ (a' b' c' d' e' : ℕ),
        n ∉ {x : ℕ | a' ∣ x ∨ b' ∣ x ∨ c' ∣ x ∨ d' ∣ x ∨ e' ∣ x} ∧
        n + a' + b' + c' + d' + e' = 200 :=
by sorry

end largest_initial_number_l1194_119429


namespace middle_card_is_five_l1194_119480

/-- Represents a triple of distinct positive integers in ascending order --/
structure CardTriple where
  left : Nat
  middle : Nat
  right : Nat
  distinct : left < middle ∧ middle < right
  sum_20 : left + middle + right = 20

/-- Predicate for Bella's statement --/
def bella_cant_determine (t : CardTriple) : Prop :=
  ∃ t' : CardTriple, t'.left = t.left ∧ t' ≠ t

/-- Predicate for Della's statement --/
def della_cant_determine (t : CardTriple) : Prop :=
  ∃ t' : CardTriple, t'.middle = t.middle ∧ t' ≠ t

/-- Predicate for Nella's statement --/
def nella_cant_determine (t : CardTriple) : Prop :=
  ∃ t' : CardTriple, t'.right = t.right ∧ t' ≠ t

/-- The main theorem --/
theorem middle_card_is_five :
  ∀ t : CardTriple,
    bella_cant_determine t →
    della_cant_determine t →
    nella_cant_determine t →
    t.middle = 5 := by
  sorry

end middle_card_is_five_l1194_119480


namespace equation_holds_except_two_values_l1194_119407

theorem equation_holds_except_two_values (a : ℝ) (ha : a ≠ 0) :
  ∀ y : ℝ, y ≠ a → y ≠ -a →
  (a / (a + y) + y / (a - y)) / (y / (a + y) - a / (a - y)) = -1 :=
by sorry

end equation_holds_except_two_values_l1194_119407


namespace largest_inscribable_rectangle_area_l1194_119484

/-- The area of the largest inscribable rectangle between two congruent equilateral triangles
    within a rectangle of width 8 and length 12 -/
theorem largest_inscribable_rectangle_area
  (width : ℝ) (length : ℝ)
  (h_width : width = 8)
  (h_length : length = 12)
  (triangle_side : ℝ)
  (h_triangle_side : triangle_side = 8 * Real.sqrt 3 / 3)
  (inscribed_height : ℝ)
  (h_inscribed_height : inscribed_height = width - triangle_side)
  : inscribed_height * length = 96 - 32 * Real.sqrt 3 := by
  sorry

end largest_inscribable_rectangle_area_l1194_119484


namespace min_attacking_pairs_8x8_16rooks_l1194_119462

/-- Represents a chessboard configuration with rooks -/
structure ChessboardWithRooks where
  size : Nat
  num_rooks : Nat
  rook_positions : List (Nat × Nat)
  different_squares : rook_positions.length = num_rooks ∧ 
                      rook_positions.Nodup

/-- Counts the number of pairs of rooks that can attack each other -/
def count_attacking_pairs (board : ChessboardWithRooks) : Nat :=
  sorry

/-- Theorem stating the minimum number of attacking pairs for a specific configuration -/
theorem min_attacking_pairs_8x8_16rooks :
  ∀ (board : ChessboardWithRooks),
    board.size = 8 ∧ 
    board.num_rooks = 16 →
    count_attacking_pairs board ≥ 16 :=
  sorry

end min_attacking_pairs_8x8_16rooks_l1194_119462


namespace rabbit_weight_l1194_119438

theorem rabbit_weight (k r p : ℝ) 
  (total_weight : k + r + p = 39)
  (rabbit_parrot_weight : r + p = 3 * k)
  (rabbit_kitten_weight : r + k = 1.5 * p) :
  r = 13.65 := by
sorry

end rabbit_weight_l1194_119438


namespace solid_surface_area_theorem_l1194_119432

def solid_surface_area (s : ℝ) (h : ℝ) : ℝ :=
  let base_area := s^2
  let upper_area := 3 * s^2
  let trapezoid_area := 2 * (s + 3*s) * h
  base_area + upper_area + trapezoid_area

theorem solid_surface_area_theorem :
  solid_surface_area (4 * Real.sqrt 2) (3 * Real.sqrt 2) = 320 := by
  sorry

end solid_surface_area_theorem_l1194_119432
