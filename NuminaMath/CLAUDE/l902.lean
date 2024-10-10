import Mathlib

namespace jane_crayons_jane_crayons_proof_l902_90212

/-- Proves that Jane ends up with 80 crayons after starting with 87 and losing 7 to a hippopotamus. -/
theorem jane_crayons : ℕ → ℕ → ℕ → Prop :=
  fun initial_crayons eaten_crayons final_crayons =>
    initial_crayons = 87 ∧ 
    eaten_crayons = 7 ∧ 
    final_crayons = initial_crayons - eaten_crayons →
    final_crayons = 80

/-- The proof of the theorem. -/
theorem jane_crayons_proof : jane_crayons 87 7 80 := by
  sorry

end jane_crayons_jane_crayons_proof_l902_90212


namespace red_cars_count_l902_90225

/-- Represents the car rental problem --/
structure CarRental where
  num_white_cars : ℕ
  white_car_cost : ℕ
  red_car_cost : ℕ
  rental_duration : ℕ
  total_earnings : ℕ

/-- Calculates the number of red cars given the rental information --/
def calculate_red_cars (rental : CarRental) : ℕ :=
  (rental.total_earnings - rental.num_white_cars * rental.white_car_cost * rental.rental_duration) /
  (rental.red_car_cost * rental.rental_duration)

/-- Theorem stating that the number of red cars is 3 --/
theorem red_cars_count (rental : CarRental)
  (h1 : rental.num_white_cars = 2)
  (h2 : rental.white_car_cost = 2)
  (h3 : rental.red_car_cost = 3)
  (h4 : rental.rental_duration = 180)
  (h5 : rental.total_earnings = 2340) :
  calculate_red_cars rental = 3 := by
  sorry

#eval calculate_red_cars { num_white_cars := 2, white_car_cost := 2, red_car_cost := 3, rental_duration := 180, total_earnings := 2340 }

end red_cars_count_l902_90225


namespace tanners_savings_l902_90243

/-- Tanner's savings problem -/
theorem tanners_savings (september : ℕ) (october : ℕ) (november : ℕ) (spent : ℕ) (left : ℕ) : 
  september = 17 → 
  october = 48 → 
  spent = 49 → 
  left = 41 → 
  september + october + november - spent = left → 
  november = 25 := by
sorry

end tanners_savings_l902_90243


namespace frog_jump_distance_l902_90287

/-- The jumping contest between a grasshopper and a frog -/
theorem frog_jump_distance (grasshopper_jump : ℕ) (additional_distance : ℕ) 
  (h1 : grasshopper_jump = 25)
  (h2 : additional_distance = 15) : 
  grasshopper_jump + additional_distance = 40 := by
  sorry

#check frog_jump_distance

end frog_jump_distance_l902_90287


namespace intersection_condition_l902_90220

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Define the line
def line (k x y : ℝ) : Prop := y = k * x - 1

-- Define the intersection condition
def always_intersects (k : ℝ) : Prop :=
  ∀ x y : ℝ, hyperbola x y ∧ line k x y → x^2 - (k*x - 1)^2 = 4

-- State the theorem
theorem intersection_condition (k : ℝ) :
  always_intersects k ↔ (k = 1 ∨ k = -1 ∨ (-Real.sqrt 5 / 2 ≤ k ∧ k ≤ Real.sqrt 5 / 2)) :=
sorry

end intersection_condition_l902_90220


namespace quadratic_roots_nature_l902_90273

theorem quadratic_roots_nature (x : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*x*Real.sqrt 2 + 8
  let discriminant := (-4*Real.sqrt 2)^2 - 4*1*8
  discriminant = 0 ∧ ∃ r : ℝ, f r = 0 ∧ (∀ s : ℝ, f s = 0 → s = r) :=
by sorry

end quadratic_roots_nature_l902_90273


namespace f_strictly_increasing_and_symmetric_l902_90216

def f (x : ℝ) : ℝ := x^(1/3)

theorem f_strictly_increasing_and_symmetric :
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧
  (∀ x, f (-x) = -f x) :=
sorry

end f_strictly_increasing_and_symmetric_l902_90216


namespace polar_to_cartesian_l902_90260

theorem polar_to_cartesian :
  let r : ℝ := 4
  let θ : ℝ := 5 * Real.pi / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -2 * Real.sqrt 3 ∧ y = 2) := by sorry

end polar_to_cartesian_l902_90260


namespace correct_article_usage_l902_90247

-- Define the possible article types
inductive Article
  | Definite
  | Indefinite
  | NoArticle

-- Define the context of a noun
structure NounContext where
  isSpecific : Bool

-- Define the function to determine the correct article
def correctArticle (context : NounContext) : Article :=
  if context.isSpecific then Article.Definite else Article.Indefinite

-- Theorem statement
theorem correct_article_usage 
  (keyboard_context : NounContext)
  (computer_context : NounContext)
  (h1 : keyboard_context.isSpecific = true)
  (h2 : computer_context.isSpecific = false) :
  (correctArticle keyboard_context = Article.Definite) ∧
  (correctArticle computer_context = Article.Indefinite) := by
  sorry


end correct_article_usage_l902_90247


namespace escalator_problem_l902_90229

/-- The number of steps Petya counts while ascending the escalator -/
def steps_ascending : ℕ := 75

/-- The number of steps Petya counts while descending the escalator -/
def steps_descending : ℕ := 150

/-- The ratio of Petya's descending speed to ascending speed -/
def speed_ratio : ℚ := 3

/-- The speed of the escalator in steps per unit time -/
def escalator_speed : ℚ := 3/5

/-- The number of steps on the stopped escalator -/
def escalator_length : ℕ := 120

theorem escalator_problem :
  steps_ascending * (1 + escalator_speed) = 
  (steps_descending / speed_ratio) * (speed_ratio - escalator_speed) ∧
  escalator_length = steps_ascending * (1 + escalator_speed) := by
  sorry

end escalator_problem_l902_90229


namespace midpoint_property_l902_90208

/-- Given two points A and B in a 2D plane, proves that if C is the midpoint of AB,
    then 3 times the x-coordinate of C minus 2 times the y-coordinate of C equals 14. -/
theorem midpoint_property (A B C : ℝ × ℝ) : 
  A = (12, 9) → B = (4, 1) → C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  3 * C.1 - 2 * C.2 = 14 := by
  sorry

#check midpoint_property

end midpoint_property_l902_90208


namespace max_sum_of_factors_l902_90281

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 1638 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 1638 → 
  A + B + C ≤ X + Y + Z → A + B + C ≤ 126 := by
sorry

end max_sum_of_factors_l902_90281


namespace pure_imaginary_condition_l902_90256

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I * (-(2 + a) / 2) = (2 - a * Complex.I) / (1 + Complex.I)) → a = 2 :=
by sorry

end pure_imaginary_condition_l902_90256


namespace min_value_x_plus_2y_l902_90233

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - x*y = 0) :
  x + 2*y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ - x₀*y₀ = 0 ∧ x₀ + 2*y₀ = 8 :=
sorry

end min_value_x_plus_2y_l902_90233


namespace oil_depth_in_horizontal_cylindrical_tank_l902_90205

/-- Represents a horizontal cylindrical tank --/
structure HorizontalCylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Represents the oil in the tank --/
structure Oil where
  surfaceArea : ℝ

/-- Calculates the possible depths of oil in the tank --/
def oilDepths (tank : HorizontalCylindricalTank) (oil : Oil) : Set ℝ :=
  { h : ℝ | h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3 }

/-- Theorem statement --/
theorem oil_depth_in_horizontal_cylindrical_tank
  (tank : HorizontalCylindricalTank)
  (oil : Oil)
  (h_length : tank.length = 8)
  (h_diameter : tank.diameter = 4)
  (h_surface_area : oil.surfaceArea = 16) :
  ∀ h ∈ oilDepths tank oil, h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3 :=
by
  sorry

#check oil_depth_in_horizontal_cylindrical_tank

end oil_depth_in_horizontal_cylindrical_tank_l902_90205


namespace minimal_blue_chips_l902_90231

theorem minimal_blue_chips (r g b : ℕ) : 
  b ≥ r / 3 →
  b ≤ g / 4 →
  r + g ≥ 75 →
  (∀ b' : ℕ, b' ≥ r / 3 → b' ≤ g / 4 → b' ≥ b) →
  b = 11 := by
  sorry

end minimal_blue_chips_l902_90231


namespace cube_sum_and_reciprocal_l902_90245

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -6) :
  x^3 + 1/x^3 = -198 := by sorry

end cube_sum_and_reciprocal_l902_90245


namespace four_students_same_group_probability_l902_90210

/-- The number of students in the school -/
def total_students : ℕ := 720

/-- The number of lunch groups -/
def num_groups : ℕ := 4

/-- The size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- The probability of a student being assigned to a specific group -/
def prob_assigned_to_group : ℚ := 1 / num_groups

/-- The probability of four specific students being assigned to the same lunch group -/
def prob_four_students_same_group : ℚ := prob_assigned_to_group ^ 3

theorem four_students_same_group_probability :
  prob_four_students_same_group = 1 / 64 :=
sorry

end four_students_same_group_probability_l902_90210


namespace division_problem_l902_90297

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 1565 → 
  quotient = 65 → 
  remainder = 5 → 
  dividend = divisor * quotient + remainder → 
  divisor = 24 :=
by
  sorry

end division_problem_l902_90297


namespace distinct_triangles_in_tetrahedron_l902_90246

/-- The number of vertices in a regular tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- Calculates the number of combinations of k items chosen from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Theorem: The number of distinct triangles in a regular tetrahedron is 4 -/
theorem distinct_triangles_in_tetrahedron :
  choose tetrahedron_vertices triangle_vertices = 4 := by sorry

end distinct_triangles_in_tetrahedron_l902_90246


namespace students_favor_both_issues_l902_90218

theorem students_favor_both_issues (total : ℕ) (favor_first : ℕ) (favor_second : ℕ) (against_both : ℕ)
  (h1 : total = 500)
  (h2 : favor_first = 375)
  (h3 : favor_second = 275)
  (h4 : against_both = 40) :
  total - against_both = favor_first + favor_second - 190 := by
  sorry

end students_favor_both_issues_l902_90218


namespace can_divide_into_12_l902_90280

-- Define a circular cake
structure CircularCake where
  radius : ℝ
  center : ℝ × ℝ

-- Define a function to represent dividing a cake into equal pieces
def divide_cake (cake : CircularCake) (n : ℕ) : Set (ℝ × ℝ) :=
  sorry

-- Define our given cakes
def cake1 : CircularCake := sorry
def cake2 : CircularCake := sorry
def cake3 : CircularCake := sorry

-- State that cake1 is divided into 3 pieces
axiom cake1_division : divide_cake cake1 3

-- State that cake2 is divided into 4 pieces
axiom cake2_division : divide_cake cake2 4

-- State that all cakes have the same radius
axiom same_radius : cake1.radius = cake2.radius ∧ cake2.radius = cake3.radius

-- State that we know the center of cake3
axiom known_center3 : cake3.center = (0, 0)

-- Theorem to prove
theorem can_divide_into_12 : 
  ∃ (division : Set (ℝ × ℝ)), division = divide_cake cake3 12 :=
sorry

end can_divide_into_12_l902_90280


namespace cherry_tomato_jars_l902_90269

theorem cherry_tomato_jars (total_tomatoes : ℕ) (tomatoes_per_jar : ℕ) (h1 : total_tomatoes = 56) (h2 : tomatoes_per_jar = 8) :
  (total_tomatoes / tomatoes_per_jar : ℕ) = 7 := by
  sorry

end cherry_tomato_jars_l902_90269


namespace fourth_power_sum_l902_90232

theorem fourth_power_sum (x y z : ℝ) 
  (h1 : x + y + z = 3)
  (h2 : x^2 + y^2 + z^2 = 5)
  (h3 : x^3 + y^3 + z^3 = 7) :
  x^4 + y^4 + z^4 = 59/3 := by
sorry

end fourth_power_sum_l902_90232


namespace riley_time_outside_l902_90206

theorem riley_time_outside (D : ℝ) (jonsey_awake : ℝ) (jonsey_outside : ℝ) (riley_awake : ℝ) (inside_time : ℝ) :
  D = 24 →
  jonsey_awake = (2/3) * D →
  jonsey_outside = (1/2) * jonsey_awake →
  riley_awake = (3/4) * D →
  jonsey_awake - jonsey_outside + riley_awake - (riley_awake * (8/9)) = inside_time →
  inside_time = 10 →
  riley_awake * (8/9) = riley_awake - (inside_time - (jonsey_awake - jonsey_outside)) :=
by sorry

end riley_time_outside_l902_90206


namespace cube_surface_area_l902_90204

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p q : Point3D) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2

/-- The vertices of the cube -/
def P : Point3D := ⟨6, 11, 11⟩
def Q : Point3D := ⟨7, 7, 2⟩
def R : Point3D := ⟨10, 2, 10⟩

theorem cube_surface_area : 
  squaredDistance P Q = squaredDistance P R ∧ 
  squaredDistance P R = squaredDistance Q R ∧
  squaredDistance Q R = 98 →
  (6 * ((squaredDistance P Q).sqrt / Real.sqrt 2)^2 : ℝ) = 294 := by
  sorry

end cube_surface_area_l902_90204


namespace triangle_is_isosceles_right_l902_90215

/-- A triangle with side lengths a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  h : R = (a * Real.sqrt (b * c)) / (b + c)

/-- The angles of a triangle -/
structure Angles where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Theorem: If a triangle's circumradius satisfies the given equation, 
    then it is an isosceles right triangle -/
theorem triangle_is_isosceles_right (t : Triangle) : 
  ∃ (angles : Angles), 
    angles.α = 90 ∧ 
    angles.β = 45 ∧ 
    angles.γ = 45 := by
  sorry

end triangle_is_isosceles_right_l902_90215


namespace min_shift_for_symmetry_l902_90230

theorem min_shift_for_symmetry (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = Real.sqrt 3 * Real.cos x + Real.sin x) →
  m > 0 →
  (∀ x, f (x + m) = f (-x + m)) →
  m ≥ π / 6 :=
sorry

end min_shift_for_symmetry_l902_90230


namespace arithmetic_mean_of_special_set_l902_90288

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 2) :
  let set := [1 - 2 / n, 1 - 2 / n] ++ List.replicate (n - 2) 1
  (set.sum / n : ℚ) = 1 - 4 / n^2 := by sorry

end arithmetic_mean_of_special_set_l902_90288


namespace constant_d_value_l902_90265

theorem constant_d_value (e c f : ℝ) : 
  (∃ d : ℝ, ∀ x : ℝ, 
    (3 * x^3 - 2 * x^2 + x - 5/4) * (e * x^3 + d * x^2 + c * x + f) = 
    9 * x^6 - 5 * x^5 - x^4 + 20 * x^3 - 25/4 * x^2 + 15/4 * x - 5/2) →
  (∃ d : ℝ, d = 1/3) := by
sorry

end constant_d_value_l902_90265


namespace inequality_proof_l902_90290

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (1 + 2 * a) + 1 / (1 + 2 * b) + 1 / (1 + 2 * c) ≥ 1 := by
  sorry

end inequality_proof_l902_90290


namespace women_in_room_l902_90293

theorem women_in_room (x : ℕ) (h1 : 4 * x + 2 = 14) : 2 * (5 * x - 3) = 24 := by
  sorry

end women_in_room_l902_90293


namespace complement_B_union_A_equals_open_interval_l902_90292

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 3*x < 4}

-- Define set B
def B : Set ℝ := {x : ℝ | |x| ≥ 2}

-- Theorem statement
theorem complement_B_union_A_equals_open_interval :
  (Set.compl B) ∪ A = Set.Ioo (-2 : ℝ) 4 :=
sorry

end complement_B_union_A_equals_open_interval_l902_90292


namespace negation_equivalence_l902_90278

theorem negation_equivalence :
  (¬ ∀ m : ℝ, m > 0 → m^2 > 0) ↔ (∃ m : ℝ, m ≤ 0 ∧ m^2 ≤ 0) :=
by sorry

end negation_equivalence_l902_90278


namespace prism_volume_given_tangent_sphere_l902_90262

-- Define the sphere
structure Sphere where
  volume : ℝ

-- Define the right triangular prism
structure RightTriangularPrism where
  baseEdgeLength : ℝ
  height : ℝ

-- Define the property of the sphere being tangent to the prism
def isTangentTo (s : Sphere) (p : RightTriangularPrism) : Prop :=
  ∃ (r : ℝ), s.volume = (4/3) * Real.pi * r^3 ∧
             p.baseEdgeLength = 2 * Real.sqrt 3 * r ∧
             p.height = 2 * r

-- Theorem statement
theorem prism_volume_given_tangent_sphere (s : Sphere) (p : RightTriangularPrism) :
  s.volume = 9 * Real.pi / 2 →
  isTangentTo s p →
  (Real.sqrt 3 / 4) * p.baseEdgeLength^2 * p.height = 81 * Real.sqrt 3 / 4 :=
by sorry

end prism_volume_given_tangent_sphere_l902_90262


namespace quadratic_equations_solutions_l902_90253

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 - 4*x + 1 = 0
  let eq2 : ℝ → Prop := λ x ↦ x^2 - 5*x + 6 = 0
  let sol1 : Set ℝ := {2 + Real.sqrt 3, 2 - Real.sqrt 3}
  let sol2 : Set ℝ := {2, 3}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ x, eq1 x → x ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ x, eq2 x → x ∈ sol2) :=
by sorry

end quadratic_equations_solutions_l902_90253


namespace older_child_age_l902_90296

def mother_charge : ℚ := 6.5
def child_charge_per_year : ℚ := 0.5
def total_bill : ℚ := 14.5
def num_children : ℕ := 4

def is_valid_age (triplet_age : ℕ) (older_age : ℕ) : Prop :=
  triplet_age > 0 ∧ 
  older_age > triplet_age ∧
  mother_charge + child_charge_per_year * (3 * triplet_age + older_age) = total_bill

theorem older_child_age :
  ∃ (triplet_age : ℕ) (older_age : ℕ), 
    is_valid_age triplet_age older_age ∧
    (older_age = 4 ∨ older_age = 7) ∧
    ¬∃ (other_age : ℕ), other_age ≠ 4 ∧ other_age ≠ 7 ∧ is_valid_age triplet_age other_age :=
by sorry

end older_child_age_l902_90296


namespace p_sufficient_not_necessary_for_q_l902_90274

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, a < x^2 + 1
def q (a : ℝ) : Prop := ∃ x₀ : ℝ, a < 3 - x₀^2

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬(p a)) :=
sorry

end p_sufficient_not_necessary_for_q_l902_90274


namespace two_by_one_cuboid_net_l902_90266

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of unit squares in the net of a cuboid -/
def net_squares (c : Cuboid) : ℕ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

/-- Theorem: A 2x1x1 cuboid's net has 10 unit squares, and removing any one leaves 9 -/
theorem two_by_one_cuboid_net :
  let c : Cuboid := ⟨2, 1, 1⟩
  net_squares c = 10 ∧ net_squares c - 1 = 9 := by
  sorry

#eval net_squares ⟨2, 1, 1⟩

end two_by_one_cuboid_net_l902_90266


namespace fraction_subtraction_l902_90282

theorem fraction_subtraction : 
  (((2 + 4 + 6 + 8) : ℚ) / (1 + 3 + 5 + 7)) - ((1 + 3 + 5 + 7) / (2 + 4 + 6 + 8)) = 9 / 20 := by
  sorry

end fraction_subtraction_l902_90282


namespace root_implies_a_value_l902_90223

theorem root_implies_a_value (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x = 0 ∧ x = 1) → a = -1 := by
  sorry

end root_implies_a_value_l902_90223


namespace class_size_l902_90285

theorem class_size (error_increase : ℝ) (average_increase : ℝ) (n : ℕ) : 
  error_increase = 20 →
  average_increase = 1/2 →
  error_increase = n * average_increase →
  n = 40 := by
  sorry

end class_size_l902_90285


namespace combined_salaries_l902_90200

theorem combined_salaries (average_salary : ℕ) (b_salary : ℕ) (total_people : ℕ) :
  average_salary = 8200 →
  b_salary = 5000 →
  total_people = 5 →
  (average_salary * total_people) - b_salary = 36000 :=
by sorry

end combined_salaries_l902_90200


namespace negation_of_existence_negation_of_cubic_equation_l902_90268

theorem negation_of_existence (f : ℝ → ℝ) : 
  (¬ ∃ x : ℝ, f x = 0) ↔ (∀ x : ℝ, f x ≠ 0) := by sorry

theorem negation_of_cubic_equation :
  (¬ ∃ x : ℝ, x^3 + 5*x - 2 = 0) ↔ (∀ x : ℝ, x^3 + 5*x - 2 ≠ 0) := by
  apply negation_of_existence (λ x => x^3 + 5*x - 2)

end negation_of_existence_negation_of_cubic_equation_l902_90268


namespace hyperbola_midpoint_exists_l902_90203

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

-- Define the midpoint
def is_midpoint (x1 y1 x2 y2 x0 y0 : ℝ) : Prop :=
  x0 = (x1 + x2) / 2 ∧ y0 = (y1 + y2) / 2

theorem hyperbola_midpoint_exists :
  ∃ (x1 y1 x2 y2 : ℝ),
    is_on_hyperbola x1 y1 ∧
    is_on_hyperbola x2 y2 ∧
    is_midpoint x1 y1 x2 y2 (-1) (-4) :=
by sorry

end hyperbola_midpoint_exists_l902_90203


namespace cos_180_deg_l902_90277

/-- The cosine of an angle in degrees -/
noncomputable def cos_deg (θ : ℝ) : ℝ := 
  (Complex.exp (θ * Complex.I * Real.pi / 180)).re

/-- Theorem: The cosine of 180 degrees is -1 -/
theorem cos_180_deg : cos_deg 180 = -1 := by sorry

end cos_180_deg_l902_90277


namespace die_roll_probabilities_l902_90271

theorem die_roll_probabilities :
  let n : ℕ := 7  -- number of rolls
  let p : ℝ := 1/6  -- probability of rolling a 4
  let q : ℝ := 1 - p  -- probability of not rolling a 4

  -- (a) Probability of rolling at least one 4 in 7 rolls
  let prob_at_least_one : ℝ := 1 - q^n

  -- (b) Probability of rolling exactly one 4 in 7 rolls
  let prob_exactly_one : ℝ := n * p * q^(n-1)

  -- (c) Probability of rolling at most one 4 in 7 rolls
  let prob_at_most_one : ℝ := q^n + n * p * q^(n-1)

  -- Prove that the calculated probabilities are correct
  (prob_at_least_one = 1 - (5/6)^7) ∧
  (prob_exactly_one = 7 * (1/6) * (5/6)^6) ∧
  (prob_at_most_one = (5/6)^7 + 7 * (1/6) * (5/6)^6) :=
by
  sorry

end die_roll_probabilities_l902_90271


namespace binomial_150_150_l902_90257

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end binomial_150_150_l902_90257


namespace max_m_value_l902_90242

theorem max_m_value (b a m : ℝ) (hb : b > 0) :
  (∀ a, (b - (a - 2))^2 + (Real.log b - (a - 1))^2 ≥ m^2 - m) →
  m ≤ 2 :=
by sorry

end max_m_value_l902_90242


namespace jerrys_pool_depth_l902_90258

/-- Calculates the depth of Jerry's pool given water usage constraints -/
theorem jerrys_pool_depth :
  ∀ (total_water drinking_cooking shower_water showers pool_length pool_width : ℕ),
  total_water = 1000 →
  drinking_cooking = 100 →
  shower_water = 20 →
  showers = 15 →
  pool_length = 10 →
  pool_width = 10 →
  (total_water - (drinking_cooking + shower_water * showers)) / (pool_length * pool_width) = 6 := by
  sorry

#check jerrys_pool_depth

end jerrys_pool_depth_l902_90258


namespace tan_plus_reciprocal_l902_90248

theorem tan_plus_reciprocal (θ : Real) (h : Real.sin (2 * θ) = 2/3) :
  Real.tan θ + (Real.tan θ)⁻¹ = 3 := by
  sorry

end tan_plus_reciprocal_l902_90248


namespace hospital_staff_ratio_l902_90294

theorem hospital_staff_ratio (total : ℕ) (nurses : ℕ) (doctors : ℕ) :
  total = 250 →
  nurses = 150 →
  doctors = total - nurses →
  (doctors : ℚ) / nurses = 2 / 3 := by
sorry

end hospital_staff_ratio_l902_90294


namespace interest_calculation_l902_90295

/-- Given a principal amount and number of years, proves that if simple interest
    at 5% per annum is 50 and compound interest at the same rate is 51.25,
    then the number of years is 2. -/
theorem interest_calculation (P n : ℝ) : 
  P * n / 20 = 50 →
  P * ((1 + 5/100)^n - 1) = 51.25 →
  n = 2 := by
sorry

end interest_calculation_l902_90295


namespace probability_three_blue_marbles_specific_l902_90213

/-- Represents the probability of drawing 3 blue marbles from a jar --/
def probability_three_blue_marbles (red blue yellow : ℕ) : ℚ :=
  let total := red + blue + yellow
  (blue / total) * ((blue - 1) / (total - 1)) * ((blue - 2) / (total - 2))

/-- Theorem stating the probability of drawing 3 blue marbles from a specific jar configuration --/
theorem probability_three_blue_marbles_specific :
  probability_three_blue_marbles 3 4 13 = 1 / 285 := by
  sorry

#eval probability_three_blue_marbles 3 4 13

end probability_three_blue_marbles_specific_l902_90213


namespace cardinality_of_B_l902_90251

def A : Finset Int := {-3, -2, -1, 1, 2, 3, 4}

def f (a : Int) : Int := Int.natAbs a

def B : Finset Int := Finset.image f A

theorem cardinality_of_B : Finset.card B = 4 := by
  sorry

end cardinality_of_B_l902_90251


namespace train_length_train_length_proof_l902_90276

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  let speed_ms : ℝ := train_speed * 1000 / 3600
  speed_ms * crossing_time

/-- Proof that a train with speed 48 km/hr crossing a pole in 9 seconds has a length of approximately 119.97 meters -/
theorem train_length_proof :
  ∃ ε > 0, |train_length 48 9 - 119.97| < ε :=
by
  sorry

end train_length_train_length_proof_l902_90276


namespace cubic_factorization_l902_90279

theorem cubic_factorization (x : ℝ) :
  (189 * x^3 + 129 * x^2 + 183 * x + 19 = (4*x - 2)^3 + (5*x + 3)^3) ∧
  (x^3 + 69 * x^2 + 87 * x + 167 = 5*(x + 3)^3 - 4*(x - 2)^3) := by
  sorry

end cubic_factorization_l902_90279


namespace average_sale_is_5500_l902_90286

def sales : List ℕ := [5435, 5927, 5855, 6230, 5562]
def sixth_month_sale : ℕ := 3991
def num_months : ℕ := 6

theorem average_sale_is_5500 :
  (sales.sum + sixth_month_sale) / num_months = 5500 := by
  sorry

end average_sale_is_5500_l902_90286


namespace space_line_relations_l902_90270

-- Define a type for lines in space
variable (Line : Type)

-- Define the parallel relation
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation
variable (perpendicular : Line → Line → Prop)

-- Define the intersects relation
variable (intersects : Line → Line → Prop)

-- Define a type for planes
variable (Plane : Type)

-- Define a relation for a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define three non-intersecting lines
variable (a b c : Line)

-- Define two planes
variable (α β : Plane)

-- State that the lines are non-intersecting
variable (h_non_intersect : ¬(intersects a b ∨ intersects b c ∨ intersects a c))

theorem space_line_relations :
  (∀ x y z, parallel x y → parallel y z → parallel x z) ∧
  ¬(∀ x y z, perpendicular x y → perpendicular y z → parallel x z) ∧
  ¬(∀ x y z, intersects x y → intersects y z → intersects x z) ∧
  ¬(∀ x y p q, line_in_plane x p → line_in_plane y q → x ≠ y → ¬(parallel x y ∨ intersects x y)) :=
by sorry

end space_line_relations_l902_90270


namespace like_terms_imply_value_l902_90254

theorem like_terms_imply_value (a b : ℤ) : 
  (1 = a - 1) → (b + 1 = 4) → (a - b)^2023 = -1 := by
  sorry

end like_terms_imply_value_l902_90254


namespace line_through_coefficients_l902_90234

/-- Given two lines that intersect at (2,3), prove that the line passing through their coefficients has a specific equation -/
theorem line_through_coefficients 
  (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : 2*a₁ + 3*b₁ + 1 = 0) 
  (h₂ : 2*a₂ + 3*b₂ + 1 = 0) :
  ∀ (x y : ℝ), (x = a₁ ∧ y = b₁) ∨ (x = a₂ ∧ y = b₂) → 2*x + 3*y + 1 = 0 := by
  sorry

end line_through_coefficients_l902_90234


namespace triangle_problem_l902_90298

-- Define the triangle ABC
variable (A B C : Real) -- Angles
variable (a b c : Real) -- Sides
variable (S : Real) -- Area

-- State the theorem
theorem triangle_problem 
  (h1 : 2 * Real.cos (2 * B) = 4 * Real.cos B - 3)
  (h2 : S = Real.sqrt 3)
  (h3 : a * Real.sin A + c * Real.sin C = 5 * Real.sin B) :
  B = π / 3 ∧ b = (5 + Real.sqrt 21) / 2 := by
  sorry

end triangle_problem_l902_90298


namespace abs_inequality_solution_set_l902_90227

theorem abs_inequality_solution_set :
  {x : ℝ | |x - 1| < 2} = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end abs_inequality_solution_set_l902_90227


namespace prism_volume_l902_90236

/-- Given a right rectangular prism with dimensions a, b, and c satisfying certain conditions,
    prove that its volume is 200 cubic centimeters. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 50) (h2 : b * c = 80) (h3 : a * c = 100)
    (h4 : ∃ n : ℕ, (a * c : ℝ) = n ^ 2) : a * b * c = 200 := by
  sorry

end prism_volume_l902_90236


namespace cake_recipe_difference_l902_90228

theorem cake_recipe_difference (total_flour total_sugar flour_added : ℕ) : 
  total_flour = 9 → 
  total_sugar = 11 → 
  flour_added = 4 → 
  total_sugar - (total_flour - flour_added) = 6 :=
by sorry

end cake_recipe_difference_l902_90228


namespace hoseok_number_division_l902_90267

theorem hoseok_number_division (x : ℤ) (h : x + 8 = 88) : x / 10 = 8 := by
  sorry

end hoseok_number_division_l902_90267


namespace cubic_polynomial_property_l902_90219

-- Define the polynomial Q(x)
def Q (x d e f : ℝ) : ℝ := x^3 + d*x^2 + e*x + f

-- Define the conditions
theorem cubic_polynomial_property (d e f : ℝ) :
  -- The y-intercept is 4
  Q 0 d e f = 4 →
  -- The mean of zeros, product of zeros, and sum of coefficients are equal
  -(d/3) = -f ∧ -(d/3) = 1 + d + e + f →
  -- The value of e is 11
  e = 11 := by
  sorry

end cubic_polynomial_property_l902_90219


namespace range_of_x_range_of_m_l902_90250

-- Define propositions p and q
def p (x m : ℝ) : Prop := x^2 - 3*m*x + 2*m^2 ≤ 0
def q (x : ℝ) : Prop := (x + 2)^2 < 1

-- Part 1
theorem range_of_x (x : ℝ) :
  p x (-2) ∧ q x → x ∈ Set.Ioc (-3) (-2) :=
sorry

-- Part 2
theorem range_of_m (m : ℝ) :
  m < 0 ∧ (∀ x, q x ↔ ¬p x m) →
  m ∈ Set.Iic (-3) ∪ Set.Icc (-1/2) 0 :=
sorry

end range_of_x_range_of_m_l902_90250


namespace more_girls_than_boys_l902_90202

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 42 →
  boys + girls = total_students →
  3 * girls = 4 * boys →
  girls - boys = 6 := by
sorry

end more_girls_than_boys_l902_90202


namespace lunch_break_duration_l902_90249

-- Define the painting rates and lunch break duration
variable (p : ℝ) -- Paula's painting rate (building/hour)
variable (h : ℝ) -- Combined rate of two helpers (building/hour)
variable (L : ℝ) -- Lunch break duration (hours)

-- Define the equations based on the given conditions
def monday_equation : Prop := (9 - L) * (p + h) = 0.4
def tuesday_equation : Prop := (7 - L) * h = 0.3
def wednesday_equation : Prop := (12 - L) * p = 0.3

-- Theorem statement
theorem lunch_break_duration 
  (eq1 : monday_equation p h L)
  (eq2 : tuesday_equation h L)
  (eq3 : wednesday_equation p L) :
  L = 0.5 := by sorry

end lunch_break_duration_l902_90249


namespace worker_completion_time_l902_90201

/-- Given workers x and y, where x can complete a job in 40 days,
    x works for 8 days, and y finishes the remaining work in 16 days,
    prove that y can complete the entire job alone in 20 days. -/
theorem worker_completion_time
  (x_completion_time : ℕ)
  (x_work_days : ℕ)
  (y_completion_time_for_remainder : ℕ)
  (h1 : x_completion_time = 40)
  (h2 : x_work_days = 8)
  (h3 : y_completion_time_for_remainder = 16) :
  (y_completion_time_for_remainder * x_completion_time) / 
  (x_completion_time - x_work_days) = 20 :=
by sorry

end worker_completion_time_l902_90201


namespace expression_evaluation_l902_90263

theorem expression_evaluation :
  let x : ℤ := -2
  2 * x * (x - 3) - (x - 2) * (x + 1) = 16 := by
  sorry

end expression_evaluation_l902_90263


namespace quadrilateral_with_equal_sine_sums_l902_90217

/-- A convex quadrilateral with angles α, β, γ, δ -/
structure ConvexQuadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  sum_360 : α + β + γ + δ = 360
  all_positive : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < δ

/-- Definition of a parallelogram -/
def IsParallelogram (q : ConvexQuadrilateral) : Prop :=
  q.α + q.γ = 180 ∧ q.β + q.δ = 180

/-- Definition of a trapezoid -/
def IsTrapezoid (q : ConvexQuadrilateral) : Prop :=
  q.α + q.β = 180 ∨ q.β + q.γ = 180 ∨ q.γ + q.δ = 180 ∨ q.δ + q.α = 180

theorem quadrilateral_with_equal_sine_sums (q : ConvexQuadrilateral) 
  (h : Real.sin q.α + Real.sin q.γ = Real.sin q.β + Real.sin q.δ) :
  IsParallelogram q ∨ IsTrapezoid q := by
  sorry


end quadrilateral_with_equal_sine_sums_l902_90217


namespace count_special_integers_l902_90224

def f (n : ℕ) : ℚ := (n^2 + n) / 2

def is_product_of_two_primes (q : ℚ) : Prop :=
  ∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ q = p1 * p2

theorem count_special_integers :
  (∃ S : Finset ℕ, (∀ n ∈ S, f n ≤ 1000 ∧ is_product_of_two_primes (f n)) ∧
                   (∀ n : ℕ, f n ≤ 1000 ∧ is_product_of_two_primes (f n) → n ∈ S) ∧
                   S.card = 5) :=
sorry

end count_special_integers_l902_90224


namespace solution_set_when_a_is_3_range_of_a_l902_90244

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |2*x - 1|

-- Theorem for Part I
theorem solution_set_when_a_is_3 :
  {x : ℝ | f x 3 ≤ 6} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 5/2} := by sorry

-- Theorem for Part II
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ a^2 - a - 13} = {a : ℝ | -Real.sqrt 14 ≤ a ∧ a ≤ 1 + Real.sqrt 13} := by sorry

end solution_set_when_a_is_3_range_of_a_l902_90244


namespace f_properties_l902_90252

def f (x : ℝ) : ℝ := |x| + 1

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y → x < 0 → f y < f x) ∧
  (∀ x y : ℝ, x < y → 0 < x → f x < f y) := by
  sorry

end f_properties_l902_90252


namespace even_decreasing_implies_increasing_l902_90284

-- Define the properties of the function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def IsDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem even_decreasing_implies_increasing 
  (f : ℝ → ℝ) (a b : ℝ) 
  (h_pos : 0 < a ∧ a < b) 
  (h_even : IsEven f) 
  (h_decreasing : IsDecreasingOn f a b) : 
  IsIncreasingOn f (-b) (-a) :=
by
  sorry

end even_decreasing_implies_increasing_l902_90284


namespace line_parabola_intersection_l902_90291

theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 - 1 ∧ p.2^2 = 4 * p.1) → k = 0 ∨ k = 1 := by
  sorry

end line_parabola_intersection_l902_90291


namespace average_of_multiples_of_10_l902_90264

theorem average_of_multiples_of_10 : 
  let multiples := List.filter (fun n => n % 10 = 0) (List.range 201)
  (List.sum multiples) / multiples.length = 105 := by
sorry

end average_of_multiples_of_10_l902_90264


namespace coin_sum_theorem_l902_90238

def coin_values : List Nat := [5, 10, 25, 50, 100]

def sum_three_coins (a b c : Nat) : Nat := a + b + c

def is_valid_sum (sum : Nat) : Prop :=
  ∃ (a b c : Nat), a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ sum_three_coins a b c = sum

theorem coin_sum_theorem :
  ¬(is_valid_sum 52) ∧
  (is_valid_sum 60) ∧
  (is_valid_sum 115) ∧
  (is_valid_sum 165) ∧
  (is_valid_sum 180) :=
sorry

end coin_sum_theorem_l902_90238


namespace apple_solution_l902_90299

/-- The number of apples each person has. -/
structure Apples where
  rebecca : ℕ
  jackie : ℕ
  adam : ℕ

/-- The conditions of the apple distribution problem. -/
def AppleConditions (a : Apples) : Prop :=
  a.rebecca = 2 * a.jackie ∧
  a.adam = a.jackie + 3 ∧
  a.adam = 9

/-- The solution to the apple distribution problem. -/
theorem apple_solution (a : Apples) (h : AppleConditions a) : a.jackie = 6 ∧ a.rebecca = 12 := by
  sorry


end apple_solution_l902_90299


namespace ab_not_necessary_nor_sufficient_for_a_plus_b_l902_90214

theorem ab_not_necessary_nor_sufficient_for_a_plus_b :
  ∃ (a b : ℝ), (a * b > 0 ∧ a + b ≤ 0) ∧
  ∃ (c d : ℝ), (c * d ≤ 0 ∧ c + d > 0) := by
  sorry

end ab_not_necessary_nor_sufficient_for_a_plus_b_l902_90214


namespace fireflies_that_flew_away_l902_90283

def initial_fireflies : ℕ := 3
def additional_fireflies : ℕ := 12 - 4
def remaining_fireflies : ℕ := 9

theorem fireflies_that_flew_away :
  initial_fireflies + additional_fireflies - remaining_fireflies = 2 := by
  sorry

end fireflies_that_flew_away_l902_90283


namespace cups_in_box_l902_90272

/-- Given an initial quantity of cups and a number of cups added, 
    calculate the total number of cups -/
def total_cups (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem stating that given 17 initial cups and 16 cups added, 
    the total number of cups is 33 -/
theorem cups_in_box : total_cups 17 16 = 33 := by
  sorry

end cups_in_box_l902_90272


namespace flower_shop_carnation_percentage_l902_90209

theorem flower_shop_carnation_percentage :
  let c : ℝ := 1  -- number of carnations (arbitrary non-zero value)
  let v : ℝ := c / 3  -- number of violets
  let t : ℝ := v / 4  -- number of tulips
  let r : ℝ := t  -- number of roses
  let total : ℝ := c + v + t + r  -- total number of flowers
  (c / total) * 100 = 200 / 3 :=
by sorry

end flower_shop_carnation_percentage_l902_90209


namespace triangle_cosine_law_l902_90255

theorem triangle_cosine_law (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (1/2) * Real.sqrt (a^2 * b^2 - ((a^2 + b^2 - c^2) / 2)^2)
  (∃ (C : ℝ), S = (1/2) * a * b * Real.sin C) →
  ∃ (C : ℝ), Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b) := by
sorry

end triangle_cosine_law_l902_90255


namespace shaded_area_square_with_circles_l902_90261

/-- The area of the shaded region in a square with circles at its vertices -/
theorem shaded_area_square_with_circles (square_side : ℝ) (circle_radius : ℝ) 
  (h1 : square_side = 8) (h2 : circle_radius = 3) : 
  ∃ (shaded_area : ℝ), shaded_area = square_side^2 - 12 * Real.sqrt 7 - 3 * Real.pi :=
by sorry

end shaded_area_square_with_circles_l902_90261


namespace room_tiles_count_l902_90207

/-- Calculates the total number of tiles needed for a room with given dimensions and tile specifications. -/
def total_tiles (room_length room_width border_width border_tile_size inner_tile_size : ℕ) : ℕ :=
  let border_tiles := 2 * (2 * (room_length - 2 * border_width) + 2 * (room_width - 2 * border_width)) - 8 * border_width
  let inner_area := (room_length - 2 * border_width) * (room_width - 2 * border_width)
  let inner_tiles := inner_area / (inner_tile_size * inner_tile_size)
  border_tiles + inner_tiles

/-- Theorem stating that for a 15-foot by 20-foot room with a double border of 1-foot tiles
    and the rest filled with 2-foot tiles, the total number of tiles used is 144. -/
theorem room_tiles_count :
  total_tiles 20 15 2 1 2 = 144 := by
  sorry

end room_tiles_count_l902_90207


namespace solve_star_equation_l902_90240

-- Define the ☆ operator
def star (a b : ℝ) : ℝ := a * b + a + b

-- Theorem statement
theorem solve_star_equation : 
  ∃! x : ℝ, star 3 x = -9 ∧ x = -3 :=
sorry

end solve_star_equation_l902_90240


namespace quadratic_with_complex_root_l902_90211

theorem quadratic_with_complex_root (a b c : ℝ) :
  (∀ x : ℂ, a * x^2 + b * x + c = 0 ↔ x = -1 + 2*I ∨ x = -1 - 2*I) →
  a = 1 ∧ b = 2 ∧ c = 5 :=
by sorry

end quadratic_with_complex_root_l902_90211


namespace car_trip_distance_l902_90222

theorem car_trip_distance (D : ℝ) : 
  (1/2 : ℝ) * D + (1/4 : ℝ) * ((1/2 : ℝ) * D) + 105 = D → D = 280 := by
  sorry

end car_trip_distance_l902_90222


namespace father_age_twice_marika_correct_target_year_l902_90226

/-- Marika's birth year -/
def marika_birth_year : ℕ := 1996

/-- The year when Marika's father's age was five times her age -/
def reference_year : ℕ := 2006

/-- Marika's father's age in the reference year -/
def father_age_reference : ℕ := 5 * (reference_year - marika_birth_year)

/-- The year when Marika's father's age will be twice her age -/
def target_year : ℕ := 2036

theorem father_age_twice_marika (year : ℕ) :
  year = target_year ↔
  (year - marika_birth_year) * 2 = (year - reference_year) + father_age_reference :=
by sorry

theorem correct_target_year : 
  (target_year - marika_birth_year) * 2 = (target_year - reference_year) + father_age_reference :=
by sorry

end father_age_twice_marika_correct_target_year_l902_90226


namespace abs_value_of_complex_l902_90237

theorem abs_value_of_complex (z : ℂ) : z = (1 + 2 * Complex.I) / Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end abs_value_of_complex_l902_90237


namespace max_triangles_two_lines_l902_90235

def points_on_line_a : ℕ := 5
def points_on_line_b : ℕ := 8

def triangles_type1 : ℕ := Nat.choose points_on_line_a 2 * Nat.choose points_on_line_b 1
def triangles_type2 : ℕ := Nat.choose points_on_line_a 1 * Nat.choose points_on_line_b 2

def total_triangles : ℕ := triangles_type1 + triangles_type2

theorem max_triangles_two_lines : total_triangles = 220 := by
  sorry

end max_triangles_two_lines_l902_90235


namespace alan_shell_collection_l902_90239

/-- Proves that Alan collected 48 shells given the conditions of the problem -/
theorem alan_shell_collection (laurie_shells : ℕ) (ben_ratio : ℚ) (alan_ratio : ℕ) : 
  laurie_shells = 36 → 
  ben_ratio = 1/3 → 
  alan_ratio = 4 → 
  (alan_ratio : ℚ) * ben_ratio * laurie_shells = 48 :=
by
  sorry

#check alan_shell_collection

end alan_shell_collection_l902_90239


namespace sets_equality_l902_90289

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem sets_equality : M = N := by sorry

end sets_equality_l902_90289


namespace school_bought_fifty_marker_cartons_l902_90259

/-- Represents the purchase of school supplies -/
structure SchoolSupplies where
  pencil_cartons : ℕ
  pencil_boxes_per_carton : ℕ
  pencil_box_cost : ℕ
  marker_carton_cost : ℕ
  total_spent : ℕ

/-- Calculates the number of marker cartons bought -/
def marker_cartons_bought (supplies : SchoolSupplies) : ℕ :=
  (supplies.total_spent - supplies.pencil_cartons * supplies.pencil_boxes_per_carton * supplies.pencil_box_cost) / supplies.marker_carton_cost

/-- Theorem stating that the school bought 50 cartons of markers -/
theorem school_bought_fifty_marker_cartons :
  let supplies : SchoolSupplies := {
    pencil_cartons := 20,
    pencil_boxes_per_carton := 10,
    pencil_box_cost := 2,
    marker_carton_cost := 4,
    total_spent := 600
  }
  marker_cartons_bought supplies = 50 := by
  sorry


end school_bought_fifty_marker_cartons_l902_90259


namespace fraction_simplification_l902_90275

theorem fraction_simplification (x y : ℚ) (hx : x = 4/6) (hy : y = 5/8) :
  (6*x + 8*y) / (48*x*y) = 9/20 := by sorry

end fraction_simplification_l902_90275


namespace smaller_integer_problem_l902_90221

theorem smaller_integer_problem (x y : ℤ) : y = 5 * x + 2 ∧ y - x = 26 → x = 6 := by
  sorry

end smaller_integer_problem_l902_90221


namespace profit_sharing_ratio_equal_l902_90241

/-- Represents the investment and profit calculation for two partners over a year -/
structure Investment where
  a_initial : ℕ  -- A's initial investment
  b_initial : ℕ  -- B's initial investment
  months : ℕ     -- Total number of months
  mid_months : ℕ -- Months after which A doubles investment

/-- Calculates the total capital-months for partner A -/
def capital_months_a (i : Investment) : ℕ :=
  i.a_initial * i.mid_months + (2 * i.a_initial) * (i.months - i.mid_months)

/-- Calculates the total capital-months for partner B -/
def capital_months_b (i : Investment) : ℕ :=
  i.b_initial * i.months

/-- Theorem stating that the profit-sharing ratio is 1:1 given the specific investment conditions -/
theorem profit_sharing_ratio_equal (i : Investment) 
  (h1 : i.a_initial = 3000)
  (h2 : i.b_initial = 4500)
  (h3 : i.months = 12)
  (h4 : i.mid_months = 6) :
  capital_months_a i = capital_months_b i := by
  sorry

end profit_sharing_ratio_equal_l902_90241
