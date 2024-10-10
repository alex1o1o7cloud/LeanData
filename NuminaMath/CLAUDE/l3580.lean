import Mathlib

namespace product_inspection_probabilities_l3580_358055

/-- Given a total of 10 products with 8 first-grade and 2 second-grade products,
    calculate probabilities when 2 products are randomly inspected. -/
theorem product_inspection_probabilities :
  let total_products : ℕ := 10
  let first_grade_products : ℕ := 8
  let second_grade_products : ℕ := 2
  let inspected_products : ℕ := 2

  -- Probability that both products are first-grade
  (Nat.choose first_grade_products inspected_products : ℚ) / 
  (Nat.choose total_products inspected_products : ℚ) = 28/45 ∧
  
  -- Probability that at least one product is second-grade
  1 - (Nat.choose first_grade_products inspected_products : ℚ) / 
  (Nat.choose total_products inspected_products : ℚ) = 17/45 :=
by
  sorry


end product_inspection_probabilities_l3580_358055


namespace f_neither_odd_nor_even_l3580_358078

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- Statement: f is neither odd nor even
theorem f_neither_odd_nor_even :
  (∃ x : ℝ, f (-x) ≠ f x) ∧ (∃ x : ℝ, f (-x) ≠ -f x) := by
  sorry

end f_neither_odd_nor_even_l3580_358078


namespace eight_percent_problem_l3580_358039

theorem eight_percent_problem (x : ℝ) : (8 / 100) * x = 64 → x = 800 := by
  sorry

end eight_percent_problem_l3580_358039


namespace work_completion_time_l3580_358043

/-- Given that:
  - A can do a work in 9 days
  - A and B together can do the work in 6 days
  Prove that B can do the work alone in 18 days -/
theorem work_completion_time (a b : ℝ) (ha : a = 9) (hab : 1 / a + 1 / b = 1 / 6) : b = 18 := by
  sorry

end work_completion_time_l3580_358043


namespace calculation_proof_inequality_system_solution_l3580_358010

-- Problem 1
theorem calculation_proof : |(-2)| - 2 * Real.sin (30 * π / 180) + (2023 ^ 0) = 2 := by sorry

-- Problem 2
theorem inequality_system_solution :
  (∀ x : ℝ, (3 * x - 1 > -7 ∧ 2 * x < x + 2) ↔ (-2 < x ∧ x < 2)) := by sorry

end calculation_proof_inequality_system_solution_l3580_358010


namespace perpendicular_lines_parallel_l3580_358026

-- Define the concept of a line in a plane
def Line : Type := Unit

-- Define the concept of a plane
def Plane : Type := Unit

-- Define the perpendicular relation between two lines in a plane
def perpendicular (p : Plane) (l1 l2 : Line) : Prop := sorry

-- Define the parallel relation between two lines in a plane
def parallel (p : Plane) (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_lines_parallel (p : Plane) (a b c : Line) :
  perpendicular p a c → perpendicular p b c → parallel p a b := by
  sorry

end perpendicular_lines_parallel_l3580_358026


namespace shopping_trip_cost_theorem_l3580_358050

def shopping_trip_cost (t_shirt_price : ℝ) (t_shirt_count : ℕ) 
                       (jeans_price : ℝ) (jeans_count : ℕ)
                       (socks_price : ℝ) (socks_count : ℕ)
                       (t_shirt_discount : ℝ) (jeans_discount : ℝ)
                       (sales_tax : ℝ) : ℝ :=
  let t_shirt_total := t_shirt_price * t_shirt_count
  let jeans_total := jeans_price * jeans_count
  let socks_total := socks_price * socks_count
  let t_shirt_discounted := t_shirt_total * (1 - t_shirt_discount)
  let jeans_discounted := jeans_total * (1 - jeans_discount)
  let subtotal := t_shirt_discounted + jeans_discounted + socks_total
  subtotal * (1 + sales_tax)

theorem shopping_trip_cost_theorem :
  shopping_trip_cost 9.65 12 29.95 3 4.50 5 0.15 0.10 0.08 = 217.93 := by
  sorry

end shopping_trip_cost_theorem_l3580_358050


namespace median_intersection_locus_l3580_358033

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  vertex : Point3D
  edge1 : Point3D → Prop
  edge2 : Point3D → Prop
  edge3 : Point3D → Prop

/-- The locus of median intersections in a trihedral angle -/
def medianIntersectionLocus (angle : TrihedralAngle) (A : Point3D) : Plane3D :=
  sorry

/-- Main theorem: The locus of median intersections is a plane parallel to OBC and 1/3 away from A -/
theorem median_intersection_locus 
  (angle : TrihedralAngle) 
  (A : Point3D) 
  (h1 : angle.edge1 A) :
  ∃ (plane : Plane3D),
    (medianIntersectionLocus angle A = plane) ∧ 
    (∃ (B C : Point3D), 
      angle.edge2 B ∧ 
      angle.edge3 C ∧ 
      (plane.a * B.x + plane.b * B.y + plane.c * B.z + plane.d = 0) ∧
      (plane.a * C.x + plane.b * C.y + plane.c * C.z + plane.d = 0)) ∧
    (∃ (k : ℝ), k = 1/3 ∧ 
      (plane.a * A.x + plane.b * A.y + plane.c * A.z + plane.d = k * 
       (plane.a * angle.vertex.x + plane.b * angle.vertex.y + plane.c * angle.vertex.z + plane.d))) :=
by sorry

end median_intersection_locus_l3580_358033


namespace calculation_proof_l3580_358091

theorem calculation_proof : 19 * 0.125 + 281 * (1/8) - 12.5 = 25 := by
  sorry

end calculation_proof_l3580_358091


namespace polynomial_division_theorem_l3580_358088

theorem polynomial_division_theorem (x : ℝ) : 
  (x + 5) * (x^4 - 5*x^3 + 2*x^2 + x - 19) + 105 = x^5 - 23*x^3 + 11*x^2 - 14*x + 10 := by
  sorry

end polynomial_division_theorem_l3580_358088


namespace train_interval_l3580_358024

-- Define the train route times
def northern_route_time : ℝ := 17
def southern_route_time : ℝ := 11

-- Define the average time difference between counterclockwise and clockwise trains
def train_arrival_difference : ℝ := 1.25

-- Define the commute time difference
def commute_time_difference : ℝ := 1

-- Theorem statement
theorem train_interval (p : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) 
  (hcommute : southern_route_time * p + northern_route_time * (1 - p) + 1 = 
              northern_route_time * p + southern_route_time * (1 - p))
  (htrain_diff : (1 - p) * 3 = train_arrival_difference) : 
  3 = 3 := by
  sorry

end train_interval_l3580_358024


namespace shopkeeper_oranges_l3580_358061

/-- The number of oranges bought by a shopkeeper -/
def oranges : ℕ := sorry

/-- The number of bananas bought by the shopkeeper -/
def bananas : ℕ := 400

/-- The percentage of oranges that are not rotten -/
def good_orange_percentage : ℚ := 85 / 100

/-- The percentage of bananas that are not rotten -/
def good_banana_percentage : ℚ := 95 / 100

/-- The overall percentage of fruits in good condition -/
def total_good_percentage : ℚ := 89 / 100

theorem shopkeeper_oranges :
  (good_orange_percentage * oranges + good_banana_percentage * bananas) / (oranges + bananas) = total_good_percentage ∧
  oranges = 600 := by sorry

end shopkeeper_oranges_l3580_358061


namespace mixed_solution_purity_l3580_358046

/-- Calculates the purity of a mixed solution given two initial solutions with different purities -/
theorem mixed_solution_purity
  (purity1 purity2 : ℚ)
  (volume1 volume2 : ℚ)
  (h1 : purity1 = 30 / 100)
  (h2 : purity2 = 60 / 100)
  (h3 : volume1 = 40)
  (h4 : volume2 = 20)
  (h5 : volume1 + volume2 = 60) :
  (purity1 * volume1 + purity2 * volume2) / (volume1 + volume2) = 40 / 100 := by
  sorry

#check mixed_solution_purity

end mixed_solution_purity_l3580_358046


namespace ratio_M_N_l3580_358045

theorem ratio_M_N (P Q R M N : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hN : N = 0.75 * R)
  (hR : R = 0.6 * P)
  (hP : P ≠ 0) : 
  M / N = 2 / 9 := by
sorry

end ratio_M_N_l3580_358045


namespace club_assignment_count_l3580_358054

/-- Represents a club -/
inductive Club
| LittleGrassLiteratureSociety
| StreetDanceClub
| FootballHouse
| CyclingClub

/-- Represents a student -/
inductive Student
| A
| B
| C
| D
| E

/-- A valid club assignment is a function from Student to Club -/
def ClubAssignment := Student → Club

/-- Predicate to check if a club assignment is valid -/
def isValidAssignment (assignment : ClubAssignment) : Prop :=
  (∀ c : Club, ∃ s : Student, assignment s = c) ∧
  (assignment Student.A ≠ Club.StreetDanceClub)

/-- The number of valid club assignments -/
def numValidAssignments : ℕ := sorry

theorem club_assignment_count :
  numValidAssignments = 180 := by sorry

end club_assignment_count_l3580_358054


namespace cosine_power_sum_l3580_358017

theorem cosine_power_sum (θ : ℝ) (x : ℂ) (n : ℤ) 
  (h1 : 0 < θ) (h2 : θ < π) (h3 : x + 1/x = 2 * Real.cos θ) : 
  x^n + 1/x^n = 2 * Real.cos (n * θ) := by
  sorry

end cosine_power_sum_l3580_358017


namespace empty_solution_set_implies_a_range_l3580_358002

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 2| + |x + 3| < a)) → a ∈ Set.Iic 5 := by
  sorry

end empty_solution_set_implies_a_range_l3580_358002


namespace prism_volume_l3580_358075

/-- The volume of a right rectangular prism given its face areas -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 30)
  (h2 : a * c = 40)
  (h3 : b * c = 60) :
  a * b * c = 120 * Real.sqrt 5 := by
  sorry

end prism_volume_l3580_358075


namespace nancy_and_rose_bracelets_l3580_358056

/-- The number of beads in each bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of metal beads Nancy has -/
def nancy_metal_beads : ℕ := 40

/-- The number of pearl beads Nancy has -/
def nancy_pearl_beads : ℕ := nancy_metal_beads + 20

/-- The number of crystal beads Rose has -/
def rose_crystal_beads : ℕ := 20

/-- The number of stone beads Rose has -/
def rose_stone_beads : ℕ := 2 * rose_crystal_beads

/-- The total number of beads Nancy and Rose have -/
def total_beads : ℕ := nancy_metal_beads + nancy_pearl_beads + rose_crystal_beads + rose_stone_beads

/-- The number of bracelets Nancy and Rose can make -/
def bracelets_made : ℕ := total_beads / beads_per_bracelet

theorem nancy_and_rose_bracelets : bracelets_made = 20 := by
  sorry

end nancy_and_rose_bracelets_l3580_358056


namespace red_tile_probability_l3580_358001

theorem red_tile_probability (n : ℕ) (h : n = 77) : 
  let red_tiles := (Finset.range n).filter (λ x => (x + 1) % 7 = 3)
  Finset.card red_tiles = 10 ∧ 
  (Finset.card red_tiles : ℚ) / n = 10 / 77 := by
  sorry

end red_tile_probability_l3580_358001


namespace trains_distance_before_meeting_l3580_358058

/-- The distance between two trains one hour before they meet -/
def distance_before_meeting (speed_A speed_B : ℝ) : ℝ :=
  speed_A + speed_B

theorem trains_distance_before_meeting 
  (speed_A speed_B total_distance : ℝ)
  (h1 : speed_A = 60)
  (h2 : speed_B = 40)
  (h3 : total_distance ≤ 250) :
  distance_before_meeting speed_A speed_B = 100 := by
  sorry

#check trains_distance_before_meeting

end trains_distance_before_meeting_l3580_358058


namespace ellipse_eccentricity_l3580_358037

theorem ellipse_eccentricity (a b c : ℝ) (θ : ℝ) : 
  a > b ∧ b > 0 ∧  -- conditions for ellipse
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (x-c)^2 + y^2 ≤ 1) ∧  -- circle inside ellipse
  (π/3 ≤ θ ∧ θ ≤ π/2) →  -- angle condition
  c/a = 3 - 2 * Real.sqrt 2 :=  -- eccentricity
by sorry

end ellipse_eccentricity_l3580_358037


namespace prop_1_prop_2_prop_3_prop_4_all_props_correct_l3580_358059

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Proposition 1
theorem prop_1 (m : ℝ) (a b : V) : m • (a - b) = m • a - m • b := by sorry

-- Proposition 2
theorem prop_2 (m n : ℝ) (a : V) : (m - n) • a = m • a - n • a := by sorry

-- Proposition 3
theorem prop_3 (m : ℝ) (a b : V) (h : m ≠ 0) : m • a = m • b → a = b := by sorry

-- Proposition 4
theorem prop_4 (m n : ℝ) (a : V) (h : a ≠ 0) : m • a = n • a → m = n := by sorry

-- All propositions are correct
theorem all_props_correct : 
  (∀ m : ℝ, ∀ a b : V, m • (a - b) = m • a - m • b) ∧
  (∀ m n : ℝ, ∀ a : V, (m - n) • a = m • a - n • a) ∧
  (∀ m : ℝ, ∀ a b : V, m ≠ 0 → (m • a = m • b → a = b)) ∧
  (∀ m n : ℝ, ∀ a : V, a ≠ 0 → (m • a = n • a → m = n)) := by sorry

end prop_1_prop_2_prop_3_prop_4_all_props_correct_l3580_358059


namespace sin_double_angle_l3580_358027

theorem sin_double_angle (θ : Real) (h : Real.sin θ = 3/5) : Real.sin (2 * θ) = 24/25 := by
  sorry

end sin_double_angle_l3580_358027


namespace product_of_digits_l3580_358099

def is_valid_number (a b c : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧
  a + b + c = 11 ∧
  (100 * a + 10 * b + c) % 5 = 0 ∧
  a = 2 * b

theorem product_of_digits (a b c : ℕ) :
  is_valid_number a b c → a * b * c = 40 := by
  sorry

end product_of_digits_l3580_358099


namespace brownies_calculation_l3580_358004

/-- The number of brownies in each batch -/
def brownies_per_batch : ℕ := 200

/-- The number of batches baked -/
def num_batches : ℕ := 10

/-- The fraction of brownies set aside for the bake sale -/
def bake_sale_fraction : ℚ := 3/4

/-- The fraction of remaining brownies put in a container -/
def container_fraction : ℚ := 3/5

/-- The number of brownies given out -/
def brownies_given_out : ℕ := 20

theorem brownies_calculation (b : ℕ) (h : b = brownies_per_batch) : 
  (1 - bake_sale_fraction) * (1 - container_fraction) * (b * num_batches) = brownies_given_out := by
  sorry

#check brownies_calculation

end brownies_calculation_l3580_358004


namespace evil_vile_live_l3580_358000

theorem evil_vile_live (E V I L : Nat) : 
  E ≠ 0 → V ≠ 0 → I ≠ 0 → L ≠ 0 →
  E < 10 → V < 10 → I < 10 → L < 10 →
  (1000 * E + 100 * V + 10 * I + L) % 73 = 0 →
  (1000 * V + 100 * I + 10 * L + E) % 74 = 0 →
  1000 * L + 100 * I + 10 * V + E = 5499 := by
sorry

end evil_vile_live_l3580_358000


namespace min_fraction_value_l3580_358029

theorem min_fraction_value (x y : ℝ) (hx : 3 ≤ x ∧ x ≤ 5) (hy : -5 ≤ y ∧ y ≤ -3) :
  (x + y) / x ≥ 2 / 5 := by
  sorry

end min_fraction_value_l3580_358029


namespace inequality_solution_l3580_358007

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^2 - 1) > 0 ↔ x > 3 ∨ x < -3 := by
  sorry

end inequality_solution_l3580_358007


namespace solution_value_l3580_358025

/-- Given that (a, b) is a solution to the linear equation 2x-7y=8,
    prove that the value of the algebraic expression 17-4a+14b is 1 -/
theorem solution_value (a b : ℝ) (h : 2*a - 7*b = 8) : 17 - 4*a + 14*b = 1 := by
  sorry

end solution_value_l3580_358025


namespace coefficient_of_monomial_degree_of_monomial_l3580_358041

-- Define a monomial type
structure Monomial where
  coefficient : ℤ
  x_exponent : ℕ
  y_exponent : ℕ

-- Define our specific monomial
def our_monomial : Monomial := ⟨-3, 2, 1⟩

-- Theorem for the coefficient
theorem coefficient_of_monomial :
  our_monomial.coefficient = -3 := by sorry

-- Theorem for the degree
theorem degree_of_monomial :
  our_monomial.x_exponent + our_monomial.y_exponent = 3 := by sorry

end coefficient_of_monomial_degree_of_monomial_l3580_358041


namespace expansion_distinct_terms_l3580_358086

/-- The number of distinct terms in the expansion of (a+b)(a+c+d+e+f) -/
def num_distinct_terms : ℕ := 9

/-- The first polynomial -/
def first_poly (a b : ℝ) : ℝ := a + b

/-- The second polynomial -/
def second_poly (a c d e f : ℝ) : ℝ := a + c + d + e + f

/-- Theorem stating that the number of distinct terms in the expansion is 9 -/
theorem expansion_distinct_terms 
  (a b c d e f : ℝ) : 
  num_distinct_terms = 9 := by sorry

end expansion_distinct_terms_l3580_358086


namespace union_of_M_and_N_l3580_358079

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x * (x - 1) = 0}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end union_of_M_and_N_l3580_358079


namespace part_one_part_two_l3580_358009

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ Real.cos t.A = -4/5

-- Part I
theorem part_one (t : Triangle) (h : triangle_conditions t) (ha : t.a = 4) :
  Real.sin t.B = 3/10 := by sorry

-- Part II
theorem part_two (t : Triangle) (h : triangle_conditions t) (hs : (1/2) * t.b * t.c * Real.sin t.A = 6) :
  t.a = 2 * Real.sqrt 34 ∧ t.c = 10 := by sorry

end part_one_part_two_l3580_358009


namespace not_parallel_implies_m_eq_one_perpendicular_implies_m_eq_neg_five_thirds_l3580_358005

/-- Two lines l₁ and l₂ in the plane -/
structure TwoLines (m : ℝ) where
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → Prop
  l₁_eq : ∀ x y, l₁ x y ↔ (3 + m) * x + 4 * y = 5 - 3 * m
  l₂_eq : ∀ x y, l₂ x y ↔ 2 * x + (m + 1) * y = -20

/-- Condition for two lines to be not parallel -/
def NotParallel (m : ℝ) (lines : TwoLines m) : Prop :=
  (3 + m) * (1 + m) - 4 * 2 ≠ 0

/-- Condition for two lines to be perpendicular -/
def Perpendicular (m : ℝ) (lines : TwoLines m) : Prop :=
  2 * (3 + m) + 4 * (1 + m) = 0

/-- Theorem: If the lines are not parallel, then m = 1 -/
theorem not_parallel_implies_m_eq_one (m : ℝ) (lines : TwoLines m) :
  NotParallel m lines → m = 1 := by sorry

/-- Theorem: If the lines are perpendicular, then m = -5/3 -/
theorem perpendicular_implies_m_eq_neg_five_thirds (m : ℝ) (lines : TwoLines m) :
  Perpendicular m lines → m = -5/3 := by sorry

end not_parallel_implies_m_eq_one_perpendicular_implies_m_eq_neg_five_thirds_l3580_358005


namespace inverse_function_point_l3580_358067

-- Define a monotonic function f
variable (f : ℝ → ℝ)
variable (h_mono : Monotone f)

-- Define the condition that f(x+1) passes through (-2, 1)
variable (h_point : f (-1) = 1)

-- State the theorem
theorem inverse_function_point :
  (Function.invFun f) 3 = -1 :=
sorry

end inverse_function_point_l3580_358067


namespace min_sum_squares_l3580_358060

theorem min_sum_squares (x y : ℝ) (h : (x - 2)^2 + (y - 3)^2 = 1) : 
  ∃ (z : ℝ), z = x^2 + y^2 ∧ (∀ (a b : ℝ), (a - 2)^2 + (b - 3)^2 = 1 → a^2 + b^2 ≥ z) ∧ z = 14 - 2 * Real.sqrt 13 :=
by sorry

end min_sum_squares_l3580_358060


namespace angle_relationship_indeterminate_l3580_358012

-- Define a plane
def Plane : Type := ℝ × ℝ → ℝ

-- Define a point in 3D space
def Point : Type := ℝ × ℝ × ℝ

-- Define a ray in 3D space
def Ray : Type := Point × Point

-- Function to calculate angle between two rays
def angle_between_rays : Ray → Ray → ℝ := sorry

-- Function to project a ray onto a plane
def project_ray : Ray → Plane → Ray := sorry

-- Function to check if a point is outside a plane
def is_outside_plane : Point → Plane → Prop := sorry

-- Theorem statement
theorem angle_relationship_indeterminate 
  (M : Plane) (P : Point) (r1 r2 : Ray) 
  (h_outside : is_outside_plane P M)
  (h_alpha : 0 < angle_between_rays r1 r2 ∧ angle_between_rays r1 r2 < π)
  (h_beta : 0 < angle_between_rays (project_ray r1 M) (project_ray r2 M) ∧ 
            angle_between_rays (project_ray r1 M) (project_ray r2 M) < π) :
  ¬ ∃ (R : ℝ → ℝ → Prop), 
    ∀ (α β : ℝ), 
      α = angle_between_rays r1 r2 → 
      β = angle_between_rays (project_ray r1 M) (project_ray r2 M) → 
      R α β :=
sorry

end angle_relationship_indeterminate_l3580_358012


namespace melanie_initial_dimes_l3580_358034

/-- Proves that Melanie initially had 7 dimes given the problem conditions. -/
theorem melanie_initial_dimes :
  ∀ (initial : ℕ),
  initial + 8 + 4 = 19 →
  initial = 7 :=
by
  sorry

end melanie_initial_dimes_l3580_358034


namespace income_redistribution_l3580_358049

/-- Represents the income distribution in a city --/
structure CityIncome where
  x : ℝ
  poor_income : ℝ
  middle_income : ℝ
  rich_income : ℝ
  tax_rate : ℝ

/-- Theorem stating the income redistribution after tax --/
theorem income_redistribution (c : CityIncome) 
  (h1 : c.poor_income = c.x)
  (h2 : c.middle_income = 3 * c.x)
  (h3 : c.rich_income = 6 * c.x)
  (h4 : c.poor_income + c.middle_income + c.rich_income = 100)
  (h5 : c.tax_rate = c.x^2 / 5 + c.x)
  (h6 : c.x = 10) :
  let tax_amount := c.rich_income * c.tax_rate / 100
  let poor_new := c.poor_income + 2 * tax_amount / 3
  let middle_new := c.middle_income + tax_amount / 3
  let rich_new := c.rich_income - tax_amount
  (poor_new = 22 ∧ middle_new = 36 ∧ rich_new = 42) := by
  sorry


end income_redistribution_l3580_358049


namespace divisibility_condition_l3580_358093

theorem divisibility_condition (n : ℕ+) : (n^2 + 1) ∣ (n + 1) ↔ n = 1 := by
  sorry

end divisibility_condition_l3580_358093


namespace min_value_x2_plus_y2_l3580_358065

theorem min_value_x2_plus_y2 (x y : ℝ) (h : x^2 + 2*x*y - y^2 = 7) :
  ∃ (m : ℝ), m = (7 * Real.sqrt 2) / 2 ∧ x^2 + y^2 ≥ m ∧ ∃ (x' y' : ℝ), x'^2 + 2*x'*y' - y'^2 = 7 ∧ x'^2 + y'^2 = m :=
sorry

end min_value_x2_plus_y2_l3580_358065


namespace balls_in_boxes_l3580_358018

theorem balls_in_boxes (n : ℕ) (k : ℕ) : n = 5 ∧ k = 4 → k^n = 1024 := by
  sorry

end balls_in_boxes_l3580_358018


namespace probability_diamond_then_ace_or_king_l3580_358036

/-- The number of cards in a combined deck of two standard decks -/
def total_cards : ℕ := 104

/-- The number of diamond cards in a combined deck of two standard decks -/
def diamond_cards : ℕ := 26

/-- The number of ace or king cards in a combined deck of two standard decks -/
def ace_or_king_cards : ℕ := 16

/-- The number of diamond cards that are not ace or king -/
def non_ace_king_diamond : ℕ := 22

/-- The number of diamond cards that are ace or king -/
def ace_king_diamond : ℕ := 4

theorem probability_diamond_then_ace_or_king :
  (diamond_cards * ace_or_king_cards - ace_king_diamond) / (total_cards * (total_cards - 1)) = 103 / 2678 := by
  sorry

end probability_diamond_then_ace_or_king_l3580_358036


namespace total_undeveloped_area_is_18750_l3580_358064

/-- The number of undeveloped land sections -/
def num_sections : ℕ := 5

/-- The area of each undeveloped land section in square feet -/
def area_per_section : ℕ := 3750

/-- The total area of undeveloped land in square feet -/
def total_undeveloped_area : ℕ := num_sections * area_per_section

/-- Theorem stating that the total area of undeveloped land is 18,750 square feet -/
theorem total_undeveloped_area_is_18750 : total_undeveloped_area = 18750 := by
  sorry

end total_undeveloped_area_is_18750_l3580_358064


namespace profit_is_eight_percent_l3580_358096

/-- Given a markup percentage and a discount percentage, calculate the profit percentage. -/
def profit_percentage (markup : ℝ) (discount : ℝ) : ℝ :=
  let marked_price := 1 + markup
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100

/-- Theorem stating that given a 30% markup and 16.92307692307692% discount, the profit is 8%. -/
theorem profit_is_eight_percent :
  profit_percentage 0.3 0.1692307692307692 = 8 := by
  sorry

end profit_is_eight_percent_l3580_358096


namespace factorial_10_mod_13_l3580_358051

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem factorial_10_mod_13 : factorial 10 % 13 = 6 := by sorry

end factorial_10_mod_13_l3580_358051


namespace sum_of_roots_of_special_quadratic_l3580_358092

-- Define a quadratic polynomial
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

-- State the theorem
theorem sum_of_roots_of_special_quadratic (a b c : ℝ) :
  (∀ x : ℝ, QuadraticPolynomial a b c (x^3 + 2*x) ≥ QuadraticPolynomial a b c (x^2 + 3)) →
  (b / a = -4 / 5) :=
by sorry

-- The sum of roots is -b/a, so if b/a = -4/5, then the sum of roots is 4/5

end sum_of_roots_of_special_quadratic_l3580_358092


namespace inequality_solution_l3580_358082

theorem inequality_solution (m n : ℝ) :
  (∀ x, 2 * m * x + 3 < 3 * x + n ↔
    ((2 * m - 3 > 0 ∧ x < (n - 3) / (2 * m - 3)) ∨
     (2 * m - 3 < 0 ∧ x > (n - 3) / (2 * m - 3)) ∨
     (m = 3 / 2 ∧ n > 3) ∨
     (m = 3 / 2 ∧ n ≤ 3 ∧ False))) :=
by sorry

end inequality_solution_l3580_358082


namespace min_sum_squares_l3580_358098

def S : Finset Int := {-9, -6, -3, 0, 1, 3, 6, 10}

theorem min_sum_squares (a b c d e f g h : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S) 
  (he : e ∈ S) (hf : f ∈ S) (hg : g ∈ S) (hh : h ∈ S)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
               f ≠ g ∧ f ≠ h ∧
               g ≠ h) :
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 2 ∧ 
  ∃ (a' b' c' d' e' f' g' h' : Int), 
    a' ∈ S ∧ b' ∈ S ∧ c' ∈ S ∧ d' ∈ S ∧ e' ∈ S ∧ f' ∈ S ∧ g' ∈ S ∧ h' ∈ S ∧
    a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ a' ≠ e' ∧ a' ≠ f' ∧ a' ≠ g' ∧ a' ≠ h' ∧
    b' ≠ c' ∧ b' ≠ d' ∧ b' ≠ e' ∧ b' ≠ f' ∧ b' ≠ g' ∧ b' ≠ h' ∧
    c' ≠ d' ∧ c' ≠ e' ∧ c' ≠ f' ∧ c' ≠ g' ∧ c' ≠ h' ∧
    d' ≠ e' ∧ d' ≠ f' ∧ d' ≠ g' ∧ d' ≠ h' ∧
    e' ≠ f' ∧ e' ≠ g' ∧ e' ≠ h' ∧
    f' ≠ g' ∧ f' ≠ h' ∧
    g' ≠ h' ∧
    (a' + b' + c' + d')^2 + (e' + f' + g' + h')^2 = 2 :=
by
  sorry

end min_sum_squares_l3580_358098


namespace laptop_price_increase_l3580_358032

theorem laptop_price_increase (P₀ : ℝ) : 
  let P₂ := P₀ * (1 + 0.06)^2
  P₂ > 56358 :=
by
  sorry

#check laptop_price_increase

end laptop_price_increase_l3580_358032


namespace expression_simplification_l3580_358008

theorem expression_simplification (a y : ℝ) : 
  ((1 : ℝ) * (3 * a^2 - 2 * a) + 2 * (a^2 - a + 2) = 5 * a^2 - 4 * a + 4) ∧ 
  ((2 : ℝ) * (2 * y^2 - 1/2 + 3 * y) - 2 * (y - y^2 + 1/2) = 4 * y^2 + y - 3/2) := by
  sorry

end expression_simplification_l3580_358008


namespace polynomial_evaluation_l3580_358071

theorem polynomial_evaluation : 
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 5
  x^2 + y^2 - z^2 + 3*x*y - z = -35 := by
sorry

end polynomial_evaluation_l3580_358071


namespace b_worked_five_days_l3580_358090

/-- Represents the number of days it takes for a person to complete the entire work alone -/
def total_days : ℕ := 15

/-- Represents the number of days it takes A to complete the remaining work after B leaves -/
def remaining_days : ℕ := 5

/-- Represents the fraction of work completed by one person in one day -/
def daily_work_rate : ℚ := 1 / total_days

/-- Represents the number of days B worked before leaving -/
def days_b_worked : ℕ := sorry

theorem b_worked_five_days :
  (days_b_worked : ℚ) * (2 * daily_work_rate) + remaining_days * daily_work_rate = 1 :=
sorry

end b_worked_five_days_l3580_358090


namespace equation_subtraction_result_l3580_358022

theorem equation_subtraction_result :
  let eq1 : ℝ → ℝ → ℝ := fun x y => 2*x + 5*y
  let eq2 : ℝ → ℝ → ℝ := fun x y => 2*x - 3*y
  let result : ℝ → ℝ := fun y => 8*y
  ∀ x y : ℝ, eq1 x y = 9 ∧ eq2 x y = 6 →
    result y = 3 :=
by sorry

end equation_subtraction_result_l3580_358022


namespace geometric_sequence_a3_l3580_358013

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 - a 2 = 6 →
  a 5 - a 1 = 15 →
  a 3 = 4 ∨ a 3 = -4 :=
by
  sorry

end geometric_sequence_a3_l3580_358013


namespace linear_function_m_value_l3580_358040

theorem linear_function_m_value :
  ∃! m : ℝ, m ≠ 0 ∧ (∀ x y : ℝ, y = m * x^(|m + 1|) - 2 → ∃ a b : ℝ, y = a * x + b) :=
by sorry

end linear_function_m_value_l3580_358040


namespace trigonometric_inequality_l3580_358057

theorem trigonometric_inequality (x y z : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < π / 2) :
  π / 2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z >
  Real.sin (2 * x) + Real.sin (2 * y) + Real.sin (2 * z) := by
  sorry

end trigonometric_inequality_l3580_358057


namespace reciprocal_of_negative_2023_l3580_358014

theorem reciprocal_of_negative_2023 :
  (1 : ℚ) / (-2023 : ℚ) = -(1 / 2023) := by sorry

end reciprocal_of_negative_2023_l3580_358014


namespace tangent_line_equation_l3580_358044

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 2

theorem tangent_line_equation :
  (∃ L : ℝ → ℝ, (L 0 = 0) ∧ (∀ x : ℝ, L x = 2*x) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x| < δ → |f x - L x| < ε * |x|)) ∧
  (∀ x₀ : ℝ, x₀ ≠ 0 →
    (∃ L : ℝ → ℝ, (L x₀ = f x₀) ∧ (∀ x : ℝ, L x = f' x₀ * (x - x₀) + f x₀) ∧
      (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x₀| < δ → |f x - L x| < ε * |x - x₀|)) →
    f' x₀ = -1/4) :=
sorry

end tangent_line_equation_l3580_358044


namespace alices_age_l3580_358095

theorem alices_age (alice : ℕ) (eve : ℕ) 
  (h1 : alice = 2 * eve) 
  (h2 : alice = eve + 10) : 
  alice = 20 := by
sorry

end alices_age_l3580_358095


namespace perimeter_PQR_l3580_358053

/-- Represents a triangle with three points -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- Theorem: Perimeter of PQR in the given configuration -/
theorem perimeter_PQR (ABC : Triangle)
  (P : ℝ × ℝ) (Q : ℝ × ℝ) (R : ℝ × ℝ)
  (h_AB : distance ABC.A ABC.B = 13)
  (h_BC : distance ABC.B ABC.C = 14)
  (h_CA : distance ABC.C ABC.A = 15)
  (h_P_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • ABC.B + t • ABC.C)
  (h_Q_on_CA : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • ABC.C + t • ABC.A)
  (h_R_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (1 - t) • ABC.A + t • ABC.B)
  (h_equal_perimeters : perimeter ⟨ABC.A, Q, R⟩ = perimeter ⟨ABC.B, P, R⟩ ∧
                        perimeter ⟨ABC.A, Q, R⟩ = perimeter ⟨ABC.C, P, Q⟩)
  (h_ratio : perimeter ⟨ABC.A, Q, R⟩ = 4/5 * perimeter ⟨P, Q, R⟩) :
  perimeter ⟨P, Q, R⟩ = 30 := by sorry

end perimeter_PQR_l3580_358053


namespace two_balls_different_color_weight_l3580_358083

-- Define the types for color and weight
inductive Color : Type
| Red : Color
| Blue : Color

inductive Weight : Type
| Light : Weight
| Heavy : Weight

-- Define the Ball type
structure Ball :=
  (color : Color)
  (weight : Weight)

-- Define the theorem
theorem two_balls_different_color_weight 
  (balls : Set Ball)
  (h1 : ∀ b : Ball, b ∈ balls → (b.color = Color.Red ∨ b.color = Color.Blue))
  (h2 : ∀ b : Ball, b ∈ balls → (b.weight = Weight.Light ∨ b.weight = Weight.Heavy))
  (h3 : ∃ b : Ball, b ∈ balls ∧ b.color = Color.Red)
  (h4 : ∃ b : Ball, b ∈ balls ∧ b.color = Color.Blue)
  (h5 : ∃ b : Ball, b ∈ balls ∧ b.weight = Weight.Light)
  (h6 : ∃ b : Ball, b ∈ balls ∧ b.weight = Weight.Heavy)
  : ∃ b1 b2 : Ball, b1 ∈ balls ∧ b2 ∈ balls ∧ b1.color ≠ b2.color ∧ b1.weight ≠ b2.weight :=
by
  sorry

end two_balls_different_color_weight_l3580_358083


namespace intersection_of_isosceles_and_right_angled_l3580_358006

-- Define the set of all triangles
def Triangle : Type := sorry

-- Define the property of being isosceles
def IsIsosceles (t : Triangle) : Prop := sorry

-- Define the property of being right-angled
def IsRightAngled (t : Triangle) : Prop := sorry

-- Define the set of isosceles triangles
def M : Set Triangle := {t : Triangle | IsIsosceles t}

-- Define the set of right-angled triangles
def N : Set Triangle := {t : Triangle | IsRightAngled t}

-- Define the property of being both isosceles and right-angled
def IsIsoscelesRightAngled (t : Triangle) : Prop := IsIsosceles t ∧ IsRightAngled t

-- Theorem statement
theorem intersection_of_isosceles_and_right_angled :
  M ∩ N = {t : Triangle | IsIsoscelesRightAngled t} := by sorry

end intersection_of_isosceles_and_right_angled_l3580_358006


namespace baby_grab_outcomes_l3580_358063

theorem baby_grab_outcomes (educational_items living_items entertainment_items : ℕ) 
  (h1 : educational_items = 4)
  (h2 : living_items = 3)
  (h3 : entertainment_items = 4) :
  educational_items + living_items + entertainment_items = 11 := by
  sorry

end baby_grab_outcomes_l3580_358063


namespace triangle_reflection_translation_l3580_358011

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the reflection across y-axis operation
def reflectAcrossYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

-- Define the translation upwards operation
def translateUpwards (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x, y := p.y + units }

-- Define the combined operation
def reflectAndTranslate (p : Point2D) (units : ℝ) : Point2D :=
  translateUpwards (reflectAcrossYAxis p) units

-- Theorem statement
theorem triangle_reflection_translation :
  let D : Point2D := { x := 3, y := 4 }
  let E : Point2D := { x := 5, y := 6 }
  let F : Point2D := { x := 5, y := 1 }
  let F' : Point2D := reflectAndTranslate F 3
  F'.x = -5 ∧ F'.y = 4 := by sorry

end triangle_reflection_translation_l3580_358011


namespace max_value_of_t_l3580_358074

theorem max_value_of_t (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → min a (b / (a^2 + b^2)) ≤ 1) ∧ 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ min a (b / (a^2 + b^2)) = 1) :=
sorry

end max_value_of_t_l3580_358074


namespace original_people_count_l3580_358042

theorem original_people_count (x : ℕ) : 
  (x / 2 : ℕ) = 18 → 
  x = 36 :=
by
  sorry

#check original_people_count

end original_people_count_l3580_358042


namespace total_length_is_6000_feet_l3580_358072

/-- Represents a path on a scale drawing -/
structure ScalePath where
  length : ℝ  -- length of the path on the drawing in inches
  scale : ℝ   -- scale factor (feet represented by 1 inch)

/-- Calculates the actual length of a path in feet -/
def actualLength (path : ScalePath) : ℝ := path.length * path.scale

/-- Theorem: The total length represented by two paths on a scale drawing is 6000 feet -/
theorem total_length_is_6000_feet (path1 path2 : ScalePath)
  (h1 : path1.length = 6 ∧ path1.scale = 500)
  (h2 : path2.length = 3 ∧ path2.scale = 1000) :
  actualLength path1 + actualLength path2 = 6000 := by
  sorry

end total_length_is_6000_feet_l3580_358072


namespace unique_positive_solution_l3580_358066

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x :=
by
  -- The unique solution is x = 1
  use 1
  constructor
  · -- Prove that x = 1 satisfies the equation
    sorry
  · -- Prove that x = 1 is the only positive solution
    sorry

#check unique_positive_solution

end unique_positive_solution_l3580_358066


namespace max_planes_in_hangar_l3580_358085

def hangar_length : ℕ := 300
def plane_length : ℕ := 40

theorem max_planes_in_hangar :
  (hangar_length / plane_length : ℕ) = 7 := by
  sorry

end max_planes_in_hangar_l3580_358085


namespace computer_ownership_increase_l3580_358077

/-- The percentage of families owning a personal computer in 1992 -/
def percentage_1992 : ℝ := 30

/-- The increase in the number of families owning a computer from 1992 to 1999 -/
def increase_1992_to_1999 : ℝ := 50

/-- The percentage of families owning at least one personal computer in 1999 -/
def percentage_1999 : ℝ := 45

theorem computer_ownership_increase :
  percentage_1999 = percentage_1992 * (1 + increase_1992_to_1999 / 100) := by
  sorry

end computer_ownership_increase_l3580_358077


namespace ant_expected_moves_l3580_358003

/-- Represents the possible parities of the ant's position -/
inductive Parity
  | Even
  | Odd

/-- Defines the ant's position on the coordinate plane -/
structure AntPosition :=
  (x : Parity)
  (y : Parity)

/-- Calculates the expected number of moves to reach an anthill from a given position -/
noncomputable def expectedMoves (pos : AntPosition) : ℝ :=
  match pos with
  | ⟨Parity.Even, Parity.Even⟩ => 4
  | ⟨Parity.Odd, Parity.Odd⟩ => 0
  | _ => 3

/-- The main theorem to be proved -/
theorem ant_expected_moves :
  let initialPos : AntPosition := ⟨Parity.Even, Parity.Even⟩
  expectedMoves initialPos = 4 := by sorry

end ant_expected_moves_l3580_358003


namespace ceiling_times_self_156_l3580_358019

theorem ceiling_times_self_156 :
  ∃! x : ℝ, ⌈x⌉ * x = 156 ∧ x = 12 := by sorry

end ceiling_times_self_156_l3580_358019


namespace specific_tetrahedron_volume_l3580_358052

/-- Tetrahedron ABCD with given properties -/
structure Tetrahedron where
  /-- Length of edge AB in cm -/
  ab_length : ℝ
  /-- Area of face ABC in cm² -/
  abc_area : ℝ
  /-- Area of face ABD in cm² -/
  abd_area : ℝ
  /-- Angle between faces ABC and ABD in radians -/
  face_angle : ℝ

/-- Volume of the tetrahedron in cm³ -/
def tetrahedron_volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the specific tetrahedron -/
theorem specific_tetrahedron_volume :
  ∃ t : Tetrahedron,
    t.ab_length = 5 ∧
    t.abc_area = 20 ∧
    t.abd_area = 18 ∧
    t.face_angle = π / 4 ∧
    tetrahedron_volume t = 24 * Real.sqrt 2 := by sorry

end specific_tetrahedron_volume_l3580_358052


namespace simplify_expression_l3580_358073

theorem simplify_expression (a : ℝ) (h : a < (1/4)) : 4*(4*a - 1)^2 = (1 - 4*a) := by
  sorry

end simplify_expression_l3580_358073


namespace greatest_divisor_with_remainders_l3580_358016

theorem greatest_divisor_with_remainders : 
  let a := 1657
  let b := 2037
  let r1 := 6
  let r2 := 5
  Int.gcd (a - r1) (b - r2) = 127 := by sorry

end greatest_divisor_with_remainders_l3580_358016


namespace sin_cos_sum_27_63_l3580_358020

theorem sin_cos_sum_27_63 : 
  Real.sin (27 * π / 180) * Real.cos (63 * π / 180) + 
  Real.cos (27 * π / 180) * Real.sin (63 * π / 180) = 1 := by
  sorry

end sin_cos_sum_27_63_l3580_358020


namespace inductive_reasoning_methods_l3580_358028

-- Define the type for reasoning methods
inductive ReasoningMethod
  | InferBallFromCircle
  | InferTriangleAngles
  | DeductBrokenChairs
  | InferPolygonAngles

-- Define a predicate for inductive reasoning
def isInductiveReasoning : ReasoningMethod → Prop
  | ReasoningMethod.InferBallFromCircle => False
  | ReasoningMethod.InferTriangleAngles => True
  | ReasoningMethod.DeductBrokenChairs => False
  | ReasoningMethod.InferPolygonAngles => True

-- Theorem stating which methods are inductive reasoning
theorem inductive_reasoning_methods :
  (isInductiveReasoning ReasoningMethod.InferTriangleAngles) ∧
  (isInductiveReasoning ReasoningMethod.InferPolygonAngles) ∧
  (¬ isInductiveReasoning ReasoningMethod.InferBallFromCircle) ∧
  (¬ isInductiveReasoning ReasoningMethod.DeductBrokenChairs) :=
by sorry


end inductive_reasoning_methods_l3580_358028


namespace rectangular_box_volume_l3580_358048

theorem rectangular_box_volume 
  (a b c : ℝ) 
  (edge_sum : 4 * a + 4 * b + 4 * c = 180) 
  (diagonal : a^2 + b^2 + c^2 = 25^2) : 
  a * b * c = 32125 := by
sorry

end rectangular_box_volume_l3580_358048


namespace free_fall_time_l3580_358094

/-- The time taken for an object to fall from a height of 490m, given the relationship h = 4.9t² -/
theorem free_fall_time : ∃ (t : ℝ), t > 0 ∧ 490 = 4.9 * t^2 ∧ t = 10 := by
  sorry

end free_fall_time_l3580_358094


namespace mixture_problem_l3580_358021

/-- Represents the initial ratio of liquid A to liquid B -/
def initial_ratio : ℚ := 7 / 5

/-- Represents the amount of mixture drawn off in liters -/
def drawn_off : ℚ := 9

/-- Represents the new ratio of liquid A to liquid B after refilling -/
def new_ratio : ℚ := 7 / 9

/-- Represents the initial amount of liquid A in the can -/
def initial_amount_A : ℚ := 21

theorem mixture_problem :
  ∃ (total : ℚ),
    total > 0 ∧
    initial_amount_A / (total - initial_amount_A) = initial_ratio ∧
    (initial_amount_A - (initial_amount_A / total) * drawn_off) /
    (total - initial_amount_A - ((total - initial_amount_A) / total) * drawn_off + drawn_off) = new_ratio :=
by sorry

end mixture_problem_l3580_358021


namespace isosceles_triangle_perimeter_l3580_358047

/-- Given a quadratic equation x^2 - (m+1)x + 2m = 0 where 3 is a root,
    and an isosceles triangle ABC where two sides have lengths equal to the roots of the equation,
    prove that the perimeter of the triangle is either 10 or 11. -/
theorem isosceles_triangle_perimeter (m : ℝ) :
  (3^2 - (m+1)*3 + 2*m = 0) →
  ∃ (a b : ℝ), (a^2 - (m+1)*a + 2*m = 0) ∧ (b^2 - (m+1)*b + 2*m = 0) ∧ 
  ((a + a + b = 10) ∨ (a + a + b = 11) ∨ (b + b + a = 10) ∨ (b + b + a = 11)) :=
by sorry

end isosceles_triangle_perimeter_l3580_358047


namespace calculate_fraction_l3580_358038

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, b^2 * f a = a^2 * f b

/-- The main theorem stating the result of the calculation -/
theorem calculate_fraction (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 2 ≠ 0) :
  (f 6 - f 3) / f 2 = 6.75 := by
  sorry

end calculate_fraction_l3580_358038


namespace right_triangle_from_equations_l3580_358030

theorem right_triangle_from_equations (a b c x : ℝ) :
  (∃ α : ℝ, α^2 + 2*a*α + b^2 = 0 ∧ α^2 + 2*c*α - b^2 = 0) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  a^2 = b^2 + c^2 :=
by sorry

end right_triangle_from_equations_l3580_358030


namespace parabola_properties_l3580_358089

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * (x^2 - 4*x + 3)

-- Define the theorem
theorem parabola_properties (a : ℝ) (h_a : a > 0) :
  -- 1. Axis of symmetry
  (∀ x : ℝ, parabola a (2 + x) = parabola a (2 - x)) ∧
  -- 2. When PQ = QA, C is at (0, 3)
  (∃ m : ℝ, m > 2 ∧ parabola a m = 3 ∧ 3 = m - 1 → parabola a 0 = 3) ∧
  -- 3. When PQ > QA, 3 < m < 4
  (∀ m : ℝ, m > 2 ∧ parabola a m = 3 ∧ 3 > m - 1 → 3 < m ∧ m < 4) :=
by sorry

end parabola_properties_l3580_358089


namespace zero_points_product_bound_l3580_358035

open Real

theorem zero_points_product_bound (a : ℝ) (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂)
  (h_zero₁ : Real.log x₁ = a * x₁)
  (h_zero₂ : Real.log x₂ = a * x₂) :
  x₁ * x₂ > Real.exp 2 := by
  sorry

end zero_points_product_bound_l3580_358035


namespace line_up_arrangements_l3580_358076

def number_of_people : ℕ := 5
def number_of_youngest : ℕ := 2

theorem line_up_arrangements :
  (number_of_people.factorial - 
   (number_of_youngest * (number_of_people - 1).factorial)) = 72 :=
by sorry

end line_up_arrangements_l3580_358076


namespace circle_equation_with_diameter_l3580_358069

def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

theorem circle_equation_with_diameter (x y : ℝ) :
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 16 →
  (x - (M.1 + N.1) / 2)^2 + (y - (M.2 + N.2) / 2)^2 = ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 4 ↔
  x^2 + y^2 = 4 := by
sorry

end circle_equation_with_diameter_l3580_358069


namespace x_plus_y_values_l3580_358023

theorem x_plus_y_values (x y : ℝ) 
  (hx : |x| = 2) 
  (hy : |y| = 5) 
  (hxy : x * y < 0) : 
  (x + y = -3) ∨ (x + y = 3) := by
sorry

end x_plus_y_values_l3580_358023


namespace repeating_decimal_47_l3580_358087

theorem repeating_decimal_47 : ∃ (x : ℚ), x = 47 / 99 ∧ 
  (∀ (n : ℕ), (100 * x - ⌊100 * x⌋) * 10^n = 47 / 100) := by
  sorry

end repeating_decimal_47_l3580_358087


namespace playground_insects_l3580_358068

def remaining_insects (spiders ants initial_ladybugs departed_ladybugs : ℕ) : ℕ :=
  spiders + ants + initial_ladybugs - departed_ladybugs

theorem playground_insects :
  remaining_insects 3 12 8 2 = 21 := by
  sorry

end playground_insects_l3580_358068


namespace powerjet_pump_volume_l3580_358080

/-- The Powerjet pump rate in gallons per hour -/
def pump_rate : ℝ := 350

/-- The time period in hours -/
def time_period : ℝ := 1.5

/-- The total volume of water pumped -/
def total_volume : ℝ := pump_rate * time_period

theorem powerjet_pump_volume : total_volume = 525 := by
  sorry

end powerjet_pump_volume_l3580_358080


namespace a_squared_plus_b_squared_eq_25_l3580_358097

/-- Definition of the sequence S_n -/
def S (n : ℕ) : ℕ := sorry

/-- The guessed formula for S_{2n-1} -/
def S_odd (n a b : ℕ) : ℕ := (4*n - 3) * (a*n + b)

/-- Theorem stating the relation between a, b and the sequence S -/
theorem a_squared_plus_b_squared_eq_25 (a b : ℕ) :
  S 1 = 1 ∧ S 3 = 25 ∧ (∀ n, S (2*n - 1) = S_odd n a b) →
  a^2 + b^2 = 25 := by
  sorry

end a_squared_plus_b_squared_eq_25_l3580_358097


namespace probability_two_zeros_not_adjacent_l3580_358084

/-- The number of ways to arrange n ones and k zeros in a row -/
def totalArrangements (n k : ℕ) : ℕ :=
  Nat.choose (n + k) k

/-- The number of ways to arrange n ones and k zeros in a row where the zeros are not adjacent -/
def favorableArrangements (n k : ℕ) : ℕ :=
  Nat.choose (n + 1) k

/-- The probability that k zeros are not adjacent when arranged with n ones in a row -/
def probabilityNonAdjacentZeros (n k : ℕ) : ℚ :=
  (favorableArrangements n k : ℚ) / (totalArrangements n k : ℚ)

theorem probability_two_zeros_not_adjacent :
  probabilityNonAdjacentZeros 4 2 = 2/3 := by
  sorry

end probability_two_zeros_not_adjacent_l3580_358084


namespace ball_placement_count_l3580_358081

-- Define the number of balls
def num_balls : ℕ := 3

-- Define the number of available boxes (excluding box 1)
def num_boxes : ℕ := 3

-- Theorem statement
theorem ball_placement_count : (num_boxes ^ num_balls) = 27 := by
  sorry

end ball_placement_count_l3580_358081


namespace coast_guard_overtakes_at_2_15pm_l3580_358070

/-- Represents the time of day in hours and minutes -/
structure TimeOfDay where
  hours : ℕ
  minutes : ℕ
  hValid : hours < 24 ∧ minutes < 60

/-- Represents the chase scenario -/
structure ChaseScenario where
  initialDistance : ℝ
  initialTime : TimeOfDay
  smugglerInitialSpeed : ℝ
  coastGuardSpeed : ℝ
  smugglerReducedSpeed : ℝ
  malfunctionTime : ℝ

/-- Calculates the time when the coast guard overtakes the smuggler -/
def overtakeTime (scenario : ChaseScenario) : TimeOfDay :=
  sorry

/-- The main theorem to prove -/
theorem coast_guard_overtakes_at_2_15pm
  (scenario : ChaseScenario)
  (h1 : scenario.initialDistance = 15)
  (h2 : scenario.initialTime = ⟨10, 0, sorry⟩)
  (h3 : scenario.smugglerInitialSpeed = 18)
  (h4 : scenario.coastGuardSpeed = 20)
  (h5 : scenario.smugglerReducedSpeed = 16)
  (h6 : scenario.malfunctionTime = 1) :
  overtakeTime scenario = ⟨14, 15, sorry⟩ :=
sorry

end coast_guard_overtakes_at_2_15pm_l3580_358070


namespace group1_larger_than_group2_l3580_358031

/-- A point on a circle -/
structure CirclePoint where
  angle : ℝ

/-- A convex polygon formed by points on a circle -/
structure ConvexPolygon where
  vertices : List CirclePoint
  is_convex : Bool

/-- The set of n points on the circle -/
def circle_points (n : ℕ) : List CirclePoint :=
  sorry

/-- Group 1: Polygons that include A₁ as a vertex -/
def group1 (n : ℕ) : List ConvexPolygon :=
  sorry

/-- Group 2: Polygons that do not include A₁ as a vertex -/
def group2 (n : ℕ) : List ConvexPolygon :=
  sorry

/-- Theorem: Group 1 contains more polygons than Group 2 -/
theorem group1_larger_than_group2 (n : ℕ) : 
  (group1 n).length > (group2 n).length :=
  sorry

end group1_larger_than_group2_l3580_358031


namespace lenny_video_game_spending_l3580_358015

def video_game_expenditure (initial_amount grocery_spending remaining_amount : ℕ) : ℕ :=
  initial_amount - grocery_spending - remaining_amount

theorem lenny_video_game_spending :
  video_game_expenditure 84 21 39 = 24 :=
by sorry

end lenny_video_game_spending_l3580_358015


namespace pages_written_theorem_l3580_358062

/-- Calculates the number of pages written in a year given the specified writing habits -/
def pages_written_per_year (pages_per_letter : ℕ) (num_friends : ℕ) (writing_frequency_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  pages_per_letter * num_friends * writing_frequency_per_week * weeks_per_year

/-- Proves that given the specified writing habits, the total number of pages written in a year is 624 -/
theorem pages_written_theorem :
  pages_written_per_year 3 2 2 52 = 624 := by
  sorry

end pages_written_theorem_l3580_358062
