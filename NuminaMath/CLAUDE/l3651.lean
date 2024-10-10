import Mathlib

namespace complex_equation_solution_l3651_365140

theorem complex_equation_solution (z : ℂ) (i : ℂ) :
  i * i = -1 →
  i * z = 2 + 4 * i →
  z = 4 - 2 * i :=
by sorry

end complex_equation_solution_l3651_365140


namespace quadruple_inequality_l3651_365156

theorem quadruple_inequality (a p q r : ℕ) 
  (ha : a > 1) (hp : p > 1) (hq : q > 1) (hr : r > 1)
  (hdiv_p : p ∣ a * q * r + 1)
  (hdiv_q : q ∣ a * p * r + 1)
  (hdiv_r : r ∣ a * p * q + 1) :
  a ≥ (p * q * r - 1) / (p * q + q * r + r * p) :=
sorry

end quadruple_inequality_l3651_365156


namespace prime_cube_minus_one_divisibility_l3651_365136

theorem prime_cube_minus_one_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_ge_3 : p ≥ 3) :
  30 ∣ (p^3 - 1) ↔ p ≡ 1 [MOD 15] := by
  sorry

end prime_cube_minus_one_divisibility_l3651_365136


namespace no_seven_edge_polyhedron_exists_polyhedron_with_n_edges_l3651_365151

/-- A convex polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure ConvexPolyhedron where
  -- We don't need to specify the internal structure for this problem
  mk :: -- Constructor

/-- The number of edges in a convex polyhedron -/
def num_edges (p : ConvexPolyhedron) : ℕ := sorry

/-- Theorem stating that no convex polyhedron has exactly 7 edges -/
theorem no_seven_edge_polyhedron : ¬∃ (p : ConvexPolyhedron), num_edges p = 7 := by sorry

/-- Theorem stating that for all natural numbers n ≥ 6 and n ≠ 7, there exists a convex polyhedron with n edges -/
theorem exists_polyhedron_with_n_edges (n : ℕ) (h1 : n ≥ 6) (h2 : n ≠ 7) : 
  ∃ (p : ConvexPolyhedron), num_edges p = n := by sorry

end no_seven_edge_polyhedron_exists_polyhedron_with_n_edges_l3651_365151


namespace mans_swimming_speed_l3651_365183

/-- The speed of the stream in km/h -/
def stream_speed : ℝ := 1.6666666666666667

/-- Proves that the man's swimming speed in still water is 5 km/h -/
theorem mans_swimming_speed (t : ℝ) (h : t > 0) : 
  let downstream_time := t
  let upstream_time := 2 * t
  let mans_speed : ℝ := stream_speed * 3
  upstream_time * (mans_speed - stream_speed) = downstream_time * (mans_speed + stream_speed) →
  mans_speed = 5 := by
sorry

end mans_swimming_speed_l3651_365183


namespace unique_number_with_properties_l3651_365186

def has_two_prime_factors (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = p^a * q^b

def count_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem unique_number_with_properties :
  ∃! n : ℕ, n > 0 ∧ has_two_prime_factors n ∧ count_divisors n = 6 ∧ sum_of_divisors n = 28 ∧ n = 12 := by
  sorry

end unique_number_with_properties_l3651_365186


namespace songs_deleted_l3651_365161

theorem songs_deleted (pictures : ℕ) (text_files : ℕ) (total_files : ℕ) (songs : ℕ) : 
  pictures = 2 → text_files = 7 → total_files = 17 → pictures + songs + text_files = total_files → songs = 8 := by
  sorry

end songs_deleted_l3651_365161


namespace product_factors_count_l3651_365184

/-- A natural number with exactly three factors is a perfect square of a prime. -/
def has_three_factors (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p^2

/-- The main theorem statement -/
theorem product_factors_count
  (a b c d : ℕ)
  (ha : has_three_factors a)
  (hb : has_three_factors b)
  (hc : has_three_factors c)
  (hd : has_three_factors d)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d)
  (hbc : b ≠ c) (hbd : b ≠ d)
  (hcd : c ≠ d) :
  (Nat.factors (a^3 * b^2 * c^4 * d^5)).length = 3465 :=
sorry

end product_factors_count_l3651_365184


namespace smallest_nonnegative_value_l3651_365135

theorem smallest_nonnegative_value (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ x ≤ y ∧
    Real.sqrt (2 * p : ℝ) - Real.sqrt x - Real.sqrt y ≥ 0 ∧
    ∀ (a b : ℕ), a > 0 → b > 0 → a ≤ b →
      Real.sqrt (2 * p : ℝ) - Real.sqrt a - Real.sqrt b ≥ 0 →
      Real.sqrt (2 * p : ℝ) - Real.sqrt a - Real.sqrt b ≥ 
      Real.sqrt (2 * p : ℝ) - Real.sqrt x - Real.sqrt y ∧
    x = (p - 1) / 2 ∧ y = (p + 1) / 2 :=
by sorry

end smallest_nonnegative_value_l3651_365135


namespace smallest_positive_multiple_of_45_l3651_365105

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l3651_365105


namespace sum_of_f_values_l3651_365106

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 2/3 → f x + f ((x - 1) / (3 * x - 2)) = x

/-- The main theorem stating the sum of f(0), f(1), and f(2) -/
theorem sum_of_f_values (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  f 0 + f 1 + f 2 = 87/40 := by
  sorry

end sum_of_f_values_l3651_365106


namespace problem_solution_l3651_365144

theorem problem_solution (r s : ℝ) 
  (h1 : 1 < r) 
  (h2 : r < s) 
  (h3 : 1 / r + 1 / s = 1) 
  (h4 : r * s = 15 / 4) : 
  s = (15 + Real.sqrt 15) / 8 := by
sorry

end problem_solution_l3651_365144


namespace arithmetic_expression_evaluation_l3651_365110

theorem arithmetic_expression_evaluation : 8 + 18 / 3 - 4 * 2 = 6 := by
  sorry

end arithmetic_expression_evaluation_l3651_365110


namespace pure_imaginary_modulus_l3651_365154

theorem pure_imaginary_modulus (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - 9) (m^2 + 2*m - 3)
  (Complex.re z = 0 ∧ Complex.im z ≠ 0) → Complex.abs z = 12 := by
  sorry

end pure_imaginary_modulus_l3651_365154


namespace parabola_focus_on_x_eq_one_l3651_365165

/-- A parabola is a conic section with a focus and directrix. -/
structure Parabola where
  /-- The focus of the parabola -/
  focus : ℝ × ℝ

/-- The standard form of a parabola equation -/
def standard_form (p : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y^2 = 4 * (x - p.focus.1)

/-- Theorem: For a parabola with its focus on the line x = 1, its standard equation is y^2 = 4x -/
theorem parabola_focus_on_x_eq_one (p : Parabola) 
    (h : p.focus.1 = 1) : 
    ∀ x y, standard_form p x y ↔ y^2 = 4*x := by
  sorry

end parabola_focus_on_x_eq_one_l3651_365165


namespace probability_theorem_l3651_365150

/-- Represents the contents of the magician's box -/
structure Box :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)

/-- Calculates the probability of drawing all red chips before blue and green chips -/
def probability_all_red_first (b : Box) : ℚ :=
  sorry

/-- The magician's box -/
def magicians_box : Box :=
  { red := 4, green := 3, blue := 1 }

/-- Theorem stating the probability of drawing all red chips first -/
theorem probability_theorem :
  probability_all_red_first magicians_box = 5 / 6720 := by
  sorry

end probability_theorem_l3651_365150


namespace iphone_price_reduction_l3651_365113

/-- 
Calculates the final price of an item after two consecutive price reductions.
-/
theorem iphone_price_reduction (initial_price : ℝ) 
  (first_reduction : ℝ) (second_reduction : ℝ) :
  initial_price = 1000 →
  first_reduction = 0.1 →
  second_reduction = 0.2 →
  initial_price * (1 - first_reduction) * (1 - second_reduction) = 720 := by
sorry

end iphone_price_reduction_l3651_365113


namespace books_after_donation_l3651_365116

theorem books_after_donation (boris_initial : ℕ) (cameron_initial : ℕ) 
  (boris_donation_fraction : ℚ) (cameron_donation_fraction : ℚ)
  (h1 : boris_initial = 24)
  (h2 : cameron_initial = 30)
  (h3 : boris_donation_fraction = 1/4)
  (h4 : cameron_donation_fraction = 1/3) :
  (boris_initial - boris_initial * boris_donation_fraction).floor +
  (cameron_initial - cameron_initial * cameron_donation_fraction).floor = 38 := by
sorry

end books_after_donation_l3651_365116


namespace increase_by_percentage_l3651_365197

/-- Theorem: Increasing 550 by 35% results in 742.5 -/
theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 550 → percentage = 35 → result = initial * (1 + percentage / 100) → result = 742.5 := by
  sorry

end increase_by_percentage_l3651_365197


namespace not_q_is_false_l3651_365138

theorem not_q_is_false (p q : Prop) (hp : ¬p) (hq : q) : ¬(¬q) :=
by sorry

end not_q_is_false_l3651_365138


namespace ricky_rose_distribution_l3651_365192

/-- Calculates the number of roses each person receives when Ricky distributes his roses. -/
def roses_per_person (initial_roses : ℕ) (stolen_roses : ℕ) (num_people : ℕ) : ℕ :=
  (initial_roses - stolen_roses) / num_people

/-- Theorem: Given the problem conditions, each person will receive 4 roses. -/
theorem ricky_rose_distribution : roses_per_person 40 4 9 = 4 := by
  sorry

end ricky_rose_distribution_l3651_365192


namespace carpenter_logs_needed_l3651_365102

/-- A carpenter building a house needs additional logs. -/
theorem carpenter_logs_needed
  (total_woodblocks_needed : ℕ)
  (logs_available : ℕ)
  (woodblocks_per_log : ℕ)
  (h1 : total_woodblocks_needed = 80)
  (h2 : logs_available = 8)
  (h3 : woodblocks_per_log = 5) :
  total_woodblocks_needed - logs_available * woodblocks_per_log = 8 * woodblocks_per_log :=
by sorry

end carpenter_logs_needed_l3651_365102


namespace octagon_interior_angles_sum_l3651_365174

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- The sum of the interior angles of an octagon is 1080 degrees -/
theorem octagon_interior_angles_sum :
  sum_interior_angles octagon_sides = 1080 := by
  sorry

end octagon_interior_angles_sum_l3651_365174


namespace dodecahedron_outer_rectangle_property_l3651_365104

/-- Regular dodecahedron with side length a -/
structure RegularDodecahedron (a : ℝ) where
  side_length : a > 0

/-- Point on a line outside a face of the dodecahedron -/
structure OuterPoint (a m : ℝ) where
  distance : m > 0

/-- Rectangle formed by four outer points -/
structure OuterRectangle (a m : ℝ) where
  A : OuterPoint a m
  B : OuterPoint a m
  C : OuterPoint a m
  D : OuterPoint a m

theorem dodecahedron_outer_rectangle_property 
  (a m : ℝ) 
  (d : RegularDodecahedron a) 
  (r : OuterRectangle a m) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
  y / x = (1 + Real.sqrt 5) / 2 := by
  sorry

end dodecahedron_outer_rectangle_property_l3651_365104


namespace gym_member_ratio_l3651_365147

theorem gym_member_ratio (f m : ℕ) (hf : f > 0) (hm : m > 0) :
  (35 : ℝ) * f + 30 * m = 32 * (f + m) →
  (f : ℝ) / m = 2 / 3 := by
sorry

end gym_member_ratio_l3651_365147


namespace unique_solution_value_l3651_365173

theorem unique_solution_value (k : ℝ) : 
  (∃! x : ℝ, (x + 3) / (k * x + 2) = x) ↔ k = -1/12 := by
  sorry

end unique_solution_value_l3651_365173


namespace water_left_proof_l3651_365178

def water_problem (initial_amount mother_drink father_extra sister_drink : ℝ) : Prop :=
  let father_drink := mother_drink + father_extra
  let total_consumed := mother_drink + father_drink + sister_drink
  let water_left := initial_amount - total_consumed
  water_left = 0.3

theorem water_left_proof :
  water_problem 1 0.1 0.2 0.3 := by
  sorry

end water_left_proof_l3651_365178


namespace intersection_forms_ellipse_l3651_365163

theorem intersection_forms_ellipse (a b : ℝ) (hab : a * b ≠ 0) :
  ∃ (h k r s : ℝ), ∀ (x y : ℝ),
    (a * x - y + b = 0) ∧ (b * x^2 + a * y^2 = a * b) →
    ((x - h) / r)^2 + ((y - k) / s)^2 = 1 :=
by sorry

end intersection_forms_ellipse_l3651_365163


namespace zoo_animal_count_l3651_365172

/-- The number of animals Brinley counted at the San Diego Zoo --/
theorem zoo_animal_count :
  let snakes : ℕ := 100
  let arctic_foxes : ℕ := 80
  let leopards : ℕ := 20
  let bee_eaters : ℕ := 10 * leopards
  let cheetahs : ℕ := snakes / 2
  let alligators : ℕ := 2 * (arctic_foxes + leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = 650 :=
by sorry


end zoo_animal_count_l3651_365172


namespace unique_prime_203B21_l3651_365121

/-- A function that generates a six-digit number of the form 203B21 given a single digit B -/
def generate_number (B : Nat) : Nat :=
  203000 + B * 100 + 21

/-- Predicate to check if a number is of the form 203B21 where B is a single digit -/
def is_valid_form (n : Nat) : Prop :=
  ∃ B : Nat, B < 10 ∧ n = generate_number B

theorem unique_prime_203B21 :
  ∃! n : Nat, is_valid_form n ∧ Nat.Prime n ∧ n = 203521 := by sorry

end unique_prime_203B21_l3651_365121


namespace record_deal_profit_difference_l3651_365153

/-- Calculates the difference in profit between two deals for selling records -/
theorem record_deal_profit_difference 
  (total_records : ℕ) 
  (sammy_price : ℚ) 
  (bryan_price_interested : ℚ) 
  (bryan_price_not_interested : ℚ) : 
  total_records = 200 →
  sammy_price = 4 →
  bryan_price_interested = 6 →
  bryan_price_not_interested = 1 →
  (total_records : ℚ) * sammy_price - 
  ((total_records / 2 : ℚ) * bryan_price_interested + 
   (total_records / 2 : ℚ) * bryan_price_not_interested) = 100 := by
  sorry

#check record_deal_profit_difference

end record_deal_profit_difference_l3651_365153


namespace smallest_with_digit_sum_41_plus_2021_l3651_365193

def digit_sum (n : ℕ) : ℕ := sorry

def is_smallest_with_digit_sum (N : ℕ) (sum : ℕ) : Prop :=
  digit_sum N = sum ∧ ∀ m < N, digit_sum m ≠ sum

theorem smallest_with_digit_sum_41_plus_2021 :
  ∃ N : ℕ, is_smallest_with_digit_sum N 41 ∧ digit_sum (N + 2021) = 10 := by
  sorry

end smallest_with_digit_sum_41_plus_2021_l3651_365193


namespace intersection_of_A_and_B_l3651_365117

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | x^2 - 1 ≥ 0}

theorem intersection_of_A_and_B : 
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x ≤ -1 ∨ 1 ≤ x ∧ x ≤ 2} := by sorry

end intersection_of_A_and_B_l3651_365117


namespace number_of_teachers_l3651_365100

/-- Represents the number of students at Queen Middle School -/
def total_students : ℕ := 1500

/-- Represents the number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- Represents the number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- Represents the number of students in each class -/
def students_per_class : ℕ := 25

/-- Represents the number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- Theorem stating that the number of teachers at Queen Middle School is 72 -/
theorem number_of_teachers : 
  (total_students * classes_per_student) / students_per_class / classes_per_teacher = 72 := by
  sorry

end number_of_teachers_l3651_365100


namespace tree_spacing_l3651_365169

/-- Proves that the distance between consecutive trees is 18 meters
    given a yard of 414 meters with 24 equally spaced trees. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) 
  (h1 : yard_length = 414)
  (h2 : num_trees = 24)
  (h3 : num_trees ≥ 2) :
  yard_length / (num_trees - 1) = 18 := by
sorry

end tree_spacing_l3651_365169


namespace intersection_A_B_l3651_365191

def U : Set Int := {-1, 3, 5, 7, 9}
def complement_A : Set Int := {-1, 9}
def B : Set Int := {3, 7, 9}

def A : Set Int := U \ complement_A

theorem intersection_A_B :
  A ∩ B = {3, 7} := by sorry

end intersection_A_B_l3651_365191


namespace count_triples_eq_30787_l3651_365108

/-- 
Counts the number of ordered triples (x,y,z) of non-negative integers 
satisfying x ≤ y ≤ z and x + y + z ≤ 100
-/
def count_triples : ℕ := 
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    let (x, y, z) := t
    x ≤ y ∧ y ≤ z ∧ x + y + z ≤ 100
  ) (Finset.product (Finset.range 101) (Finset.product (Finset.range 101) (Finset.range 101)))).card

theorem count_triples_eq_30787 : count_triples = 30787 := by
  sorry


end count_triples_eq_30787_l3651_365108


namespace inequality_holds_infinitely_often_l3651_365189

theorem inequality_holds_infinitely_often (a : ℕ → ℝ) 
  (h : ∀ n, a n > 0) : 
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ 1 + a n > a (n - 1) * (2 ^ (1 / n : ℝ)) :=
sorry

end inequality_holds_infinitely_often_l3651_365189


namespace complex_number_proof_l3651_365124

theorem complex_number_proof (z : ℂ) :
  (z.re = Complex.im (-Real.sqrt 2 + 7 * Complex.I)) ∧
  (z.im = Complex.re (Real.sqrt 7 * Complex.I + 5 * Complex.I^2)) →
  z = 7 - 5 * Complex.I :=
sorry

end complex_number_proof_l3651_365124


namespace river_length_l3651_365164

/-- The length of a river given Karen's paddling speed, river current speed, and time taken to paddle up the river -/
theorem river_length
  (karen_speed : ℝ)  -- Karen's paddling speed on still water
  (river_speed : ℝ)  -- River's current speed
  (time_taken : ℝ)   -- Time taken to paddle up the river
  (h1 : karen_speed = 10)  -- Karen's speed is 10 miles per hour
  (h2 : river_speed = 4)   -- River flows at 4 miles per hour
  (h3 : time_taken = 2)    -- It takes 2 hours to paddle up the river
  : (karen_speed - river_speed) * time_taken = 12 :=
by sorry

end river_length_l3651_365164


namespace blood_cells_in_first_sample_l3651_365122

theorem blood_cells_in_first_sample
  (total_cells : ℕ)
  (second_sample_cells : ℕ)
  (h1 : total_cells = 7341)
  (h2 : second_sample_cells = 3120) :
  total_cells - second_sample_cells = 4221 := by
  sorry

end blood_cells_in_first_sample_l3651_365122


namespace tangent_line_equation_l3651_365198

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the point P
def point_P : ℝ × ℝ := (3, 1)

-- Define a general circle
def general_circle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the tangent property
def is_tangent (circle1 circle2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y ∧
  ∀ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) → ¬(circle1 x' y' ∧ circle2 x' y')

-- State the theorem
theorem tangent_line_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    general_circle center radius point_P.1 point_P.2 ∧
    is_tangent (general_circle center radius) circle_C →
    ∃ (A B : ℝ × ℝ),
      circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
      ∀ (x y : ℝ), 2*x + y - 3 = 0 ↔ (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1) :=
sorry

end tangent_line_equation_l3651_365198


namespace polynomial_simplification_l3651_365103

theorem polynomial_simplification (q : ℝ) :
  (4 * q^4 + 5 * q^3 - 7 * q + 8) + (3 - 9 * q^3 + 5 * q^2 - 2 * q) =
  4 * q^4 - 4 * q^3 + 5 * q^2 - 9 * q + 11 :=
by sorry

end polynomial_simplification_l3651_365103


namespace histogram_total_area_is_one_l3651_365187

/-- A histogram representing a data distribution -/
structure Histogram where
  -- We don't need to define the internal structure of the histogram
  -- as we're only concerned with its total area property

/-- The total area of a histogram -/
def total_area (h : Histogram) : ℝ := sorry

/-- Theorem: The total area of a histogram representing a data distribution is equal to 1 -/
theorem histogram_total_area_is_one (h : Histogram) : total_area h = 1 := by
  sorry

end histogram_total_area_is_one_l3651_365187


namespace prob_black_then_red_standard_deck_l3651_365158

/-- A deck of cards with black cards, red cards, and jokers. -/
structure Deck :=
  (total : ℕ)
  (black : ℕ)
  (red : ℕ)
  (jokers : ℕ)

/-- The probability of drawing a black card first and a red card second from a given deck. -/
def prob_black_then_red (d : Deck) : ℚ :=
  (d.black : ℚ) / d.total * (d.red : ℚ) / (d.total - 1)

/-- The standard deck with 54 cards including jokers. -/
def standard_deck : Deck :=
  { total := 54
  , black := 26
  , red := 26
  , jokers := 2 }

theorem prob_black_then_red_standard_deck :
  prob_black_then_red standard_deck = 338 / 1431 := by
  sorry

end prob_black_then_red_standard_deck_l3651_365158


namespace jellybean_probability_l3651_365155

/-- The total number of jellybeans in the jar -/
def total_jellybeans : ℕ := 15

/-- The number of red jellybeans in the jar -/
def red_jellybeans : ℕ := 6

/-- The number of blue jellybeans in the jar -/
def blue_jellybeans : ℕ := 3

/-- The number of white jellybeans in the jar -/
def white_jellybeans : ℕ := 6

/-- The number of jellybeans picked -/
def picked_jellybeans : ℕ := 4

/-- The probability of picking at least 3 red jellybeans out of 4 -/
def prob_at_least_three_red : ℚ := 13 / 91

theorem jellybean_probability :
  let total_outcomes := Nat.choose total_jellybeans picked_jellybeans
  let favorable_outcomes := Nat.choose red_jellybeans 3 * Nat.choose (total_jellybeans - red_jellybeans) 1 +
                            Nat.choose red_jellybeans 4
  (favorable_outcomes : ℚ) / total_outcomes = prob_at_least_three_red :=
by sorry

end jellybean_probability_l3651_365155


namespace sector_area_l3651_365126

/-- Given a circular sector with perimeter 6 and central angle 1 radian, its area is 2. -/
theorem sector_area (r : ℝ) (h1 : r + 2 * r = 6) (h2 : 1 = 1) : r * r / 2 = 2 := by
  sorry

end sector_area_l3651_365126


namespace potato_ratio_l3651_365175

theorem potato_ratio (total : ℕ) (wedges : ℕ) (chip_wedge_diff : ℕ) :
  total = 67 →
  wedges = 13 →
  chip_wedge_diff = 436 →
  let remaining := total - wedges
  let fries := remaining / 2
  let chips := remaining / 2
  fries = chips := by sorry

end potato_ratio_l3651_365175


namespace quadratic_completing_square_l3651_365177

theorem quadratic_completing_square : ∀ x : ℝ, x^2 - 8*x + 6 = 0 ↔ (x - 4)^2 = 10 := by sorry

end quadratic_completing_square_l3651_365177


namespace smallest_natural_numbers_satisfying_equation_l3651_365137

theorem smallest_natural_numbers_satisfying_equation :
  ∃ (A B : ℕ+),
    (360 : ℝ) / ((A : ℝ) * (A : ℝ) * (A : ℝ) / (B : ℝ)) = 5 ∧
    ∀ (A' B' : ℕ+),
      (360 : ℝ) / ((A' : ℝ) * (A' : ℝ) * (A' : ℝ) / (B' : ℝ)) = 5 →
      (A ≤ A' ∧ B ≤ B') ∧
    A = 6 ∧
    B = 3 ∧
    A + B = 9 :=
by sorry

end smallest_natural_numbers_satisfying_equation_l3651_365137


namespace hyperbola_sum_l3651_365123

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 1 ∧ 
  k = 2 ∧ 
  c = Real.sqrt 50 ∧ 
  a = 4 ∧ 
  b * b = c * c - a * a → 
  h + k + a + b = 7 + Real.sqrt 34 := by
sorry

end hyperbola_sum_l3651_365123


namespace line_arrangement_count_l3651_365180

def number_of_students : ℕ := 5
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 3

theorem line_arrangement_count : 
  (number_of_students = number_of_boys + number_of_girls) →
  (number_of_boys = 2) →
  (number_of_girls = 3) →
  (∃ (arrangement_count : ℕ), 
    arrangement_count = (Nat.factorial number_of_boys) * (Nat.factorial (number_of_girls + 1)) ∧
    arrangement_count = 48) :=
by sorry

end line_arrangement_count_l3651_365180


namespace new_species_growth_pattern_l3651_365115

/-- Represents the shape of population growth --/
inductive GrowthShape
  | J -- J-shaped growth
  | S -- S-shaped growth

/-- Represents the population growth pattern over time --/
structure PopulationGrowth where
  initialShape : GrowthShape
  finalShape : GrowthShape

/-- Represents a new species entering an area --/
structure NewSpecies where
  enteredArea : Bool

/-- Theorem stating the population growth pattern for a new species --/
theorem new_species_growth_pattern (species : NewSpecies) 
  (h : species.enteredArea = true) : 
  ∃ (growth : PopulationGrowth), 
    growth.initialShape = GrowthShape.J ∧ 
    growth.finalShape = GrowthShape.S :=
  sorry

end new_species_growth_pattern_l3651_365115


namespace no_equal_divisors_for_squares_l3651_365129

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def divisors_3k_plus_1 (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => d > 0 ∧ n % d = 0 ∧ d % 3 = 1)

def divisors_3k_plus_2 (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => d > 0 ∧ n % d = 0 ∧ d % 3 = 2)

theorem no_equal_divisors_for_squares :
  ∀ n : ℕ, is_square n → (divisors_3k_plus_1 n).card ≠ (divisors_3k_plus_2 n).card :=
by sorry

end no_equal_divisors_for_squares_l3651_365129


namespace odd_binomial_coefficients_count_l3651_365112

theorem odd_binomial_coefficients_count (n : ℕ) : 
  (∃ m : ℕ, (Finset.filter (fun k => Nat.choose n k % 2 = 1) (Finset.range (n + 1))).card = 2^m) := by
  sorry

end odd_binomial_coefficients_count_l3651_365112


namespace wig_cost_calculation_l3651_365111

theorem wig_cost_calculation (plays : ℕ) (acts_per_play : ℕ) (wigs_per_act : ℕ) (cost_per_wig : ℕ) :
  plays = 2 →
  acts_per_play = 5 →
  wigs_per_act = 2 →
  cost_per_wig = 5 →
  plays * acts_per_play * wigs_per_act * cost_per_wig = 100 :=
by sorry

end wig_cost_calculation_l3651_365111


namespace linear_and_quadratic_sequences_properties_l3651_365168

def is_second_order_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, (a (n + 2) - a (n + 1)) - (a (n + 1) - a n) = d

def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

def is_local_geometric (a : ℕ → ℝ) : Prop :=
  ¬is_geometric a ∧ ∃ i j k : ℕ, i < j ∧ j < k ∧ (a j)^2 = a i * a k

theorem linear_and_quadratic_sequences_properties :
  (is_second_order_arithmetic (fun n => n : ℕ → ℝ) ∧
   is_local_geometric (fun n => n : ℕ → ℝ)) ∧
  (is_second_order_arithmetic (fun n => n^2 : ℕ → ℝ) ∧
   is_local_geometric (fun n => n^2 : ℕ → ℝ)) := by sorry

end linear_and_quadratic_sequences_properties_l3651_365168


namespace smallest_number_l3651_365132

def numbers : List ℤ := [0, -2, 1, 5]

theorem smallest_number (n : ℤ) (hn : n ∈ numbers) : -2 ≤ n := by
  sorry

#check smallest_number

end smallest_number_l3651_365132


namespace root_square_relation_l3651_365114

/-- The polynomial h(x) = x^3 + x^2 + 2x + 8 -/
def h (x : ℝ) : ℝ := x^3 + x^2 + 2*x + 8

/-- The polynomial j(x) = x^3 + bx^2 + cx + d -/
def j (b c d x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem root_square_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, j b c d x = 0 ↔ ∃ r : ℝ, h r = 0 ∧ x = r^2) →
  b = 1 ∧ c = -8 ∧ d = 32 := by
sorry

end root_square_relation_l3651_365114


namespace exists_hole_free_square_meter_l3651_365134

/-- Represents a point on the carpet -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the carpet with its dimensions and holes -/
structure Carpet where
  side_length : ℝ
  holes : Finset Point

/-- Represents a square piece that could be cut from the carpet -/
structure SquarePiece where
  bottom_left : Point
  side_length : ℝ

/-- Checks if a point is inside a square piece -/
def point_in_square (p : Point) (s : SquarePiece) : Prop :=
  s.bottom_left.x ≤ p.x ∧ p.x < s.bottom_left.x + s.side_length ∧
  s.bottom_left.y ≤ p.y ∧ p.y < s.bottom_left.y + s.side_length

/-- The main theorem to be proved -/
theorem exists_hole_free_square_meter (c : Carpet) 
    (h_side : c.side_length = 275)
    (h_holes : c.holes.card = 4) :
    ∃ (s : SquarePiece), s.side_length = 100 ∧ 
    s.bottom_left.x + s.side_length ≤ c.side_length ∧
    s.bottom_left.y + s.side_length ≤ c.side_length ∧
    ∀ (p : Point), p ∈ c.holes → ¬point_in_square p s :=
  sorry

end exists_hole_free_square_meter_l3651_365134


namespace complex_number_fourth_quadrant_l3651_365139

theorem complex_number_fourth_quadrant (z : ℂ) : 
  (z.re > 0) →  -- z is in the fourth quadrant (real part positive)
  (z.im < 0) →  -- z is in the fourth quadrant (imaginary part negative)
  (z.re + z.im = 7) →  -- sum of real and imaginary parts is 7
  (Complex.abs z = 13) →  -- magnitude of z is 13
  z = Complex.mk 12 (-5) :=  -- z equals 12 - 5i
by sorry

end complex_number_fourth_quadrant_l3651_365139


namespace odd_function_sum_property_l3651_365188

def is_odd_function (v : ℝ → ℝ) : Prop := ∀ x, v (-x) = -v x

theorem odd_function_sum_property (v : ℝ → ℝ) (a b : ℝ) 
  (h : is_odd_function v) : 
  v (-a) + v (-b) + v b + v a = 0 := by
  sorry

end odd_function_sum_property_l3651_365188


namespace intersecting_subset_exists_l3651_365131

theorem intersecting_subset_exists (X : Finset ℕ) (A : Fin 100 → Finset ℕ) 
  (h_size : X.card ≥ 4) 
  (h_subsets : ∀ i, A i ⊆ X) 
  (h_large : ∀ i, (A i).card > 3/4 * X.card) :
  ∃ Y : Finset ℕ, Y ⊆ X ∧ Y.card ≤ 4 ∧ ∀ i, (Y ∩ A i).Nonempty := by
  sorry


end intersecting_subset_exists_l3651_365131


namespace correct_snow_globes_count_l3651_365157

/-- The number of snow globes in each box of Christmas decorations -/
def snow_globes_per_box : ℕ := 5

/-- The number of pieces of tinsel in each box -/
def tinsel_per_box : ℕ := 4

/-- The number of Christmas trees in each box -/
def trees_per_box : ℕ := 1

/-- The total number of boxes distributed -/
def total_boxes : ℕ := 12

/-- The total number of decorations handed out -/
def total_decorations : ℕ := 120

/-- Theorem stating that the number of snow globes per box is correct -/
theorem correct_snow_globes_count :
  snow_globes_per_box = (total_decorations - total_boxes * (tinsel_per_box + trees_per_box)) / total_boxes :=
by sorry

end correct_snow_globes_count_l3651_365157


namespace cos_negative_300_degrees_l3651_365107

theorem cos_negative_300_degrees : Real.cos (-(300 * π / 180)) = 1 / 2 := by
  sorry

end cos_negative_300_degrees_l3651_365107


namespace salon_extra_cans_l3651_365167

/-- Represents the daily operations of a hair salon --/
structure Salon where
  customers : ℕ
  cans_bought : ℕ
  cans_per_customer : ℕ

/-- Calculates the number of extra cans of hairspray bought by the salon each day --/
def extra_cans (s : Salon) : ℕ :=
  s.cans_bought - (s.customers * s.cans_per_customer)

/-- Theorem stating that the salon buys 5 extra cans of hairspray each day --/
theorem salon_extra_cans :
  ∀ (s : Salon), s.customers = 14 ∧ s.cans_bought = 33 ∧ s.cans_per_customer = 2 →
  extra_cans s = 5 := by
  sorry

end salon_extra_cans_l3651_365167


namespace total_days_1996_to_2000_l3651_365182

/-- The number of days in a regular year -/
def regularYearDays : ℕ := 365

/-- The number of additional days in a leap year -/
def leapYearExtraDays : ℕ := 1

/-- The start year of our range -/
def startYear : ℕ := 1996

/-- The end year of our range -/
def endYear : ℕ := 2000

/-- The number of leap years in our range -/
def leapYearsCount : ℕ := 2

/-- Theorem: The total number of days from 1996 to 2000 (inclusive) is 1827 -/
theorem total_days_1996_to_2000 : 
  (endYear - startYear + 1) * regularYearDays + leapYearsCount * leapYearExtraDays = 1827 := by
  sorry

end total_days_1996_to_2000_l3651_365182


namespace equation_solutions_l3651_365142

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4) ∧
  (∀ x : ℝ, -2 * (x^3 - 1) = 18 ↔ x = -2) := by
  sorry

end equation_solutions_l3651_365142


namespace point_coordinates_wrt_origin_l3651_365133

/-- Given a point P with coordinates (2, -3), prove that its coordinates with respect to the origin are (2, -3) -/
theorem point_coordinates_wrt_origin : 
  let P : ℝ × ℝ := (2, -3)
  P = (2, -3) := by sorry

end point_coordinates_wrt_origin_l3651_365133


namespace water_tank_capacity_l3651_365118

/-- The capacity of a water tank given its filling rate and time to reach 3/4 capacity -/
theorem water_tank_capacity (fill_rate : ℝ) (time_to_three_quarters : ℝ) 
  (h1 : fill_rate = 10) 
  (h2 : time_to_three_quarters = 300) : 
  fill_rate * time_to_three_quarters / (3/4) = 4000 := by
  sorry

end water_tank_capacity_l3651_365118


namespace theft_loss_calculation_l3651_365179

/-- Represents the percentage of profit taken by the shopkeeper -/
def profit_percentage : ℝ := 10

/-- Represents the overall loss percentage -/
def overall_loss_percentage : ℝ := 23

/-- Represents the percentage of goods lost during theft -/
def theft_loss_percentage : ℝ := 30

/-- Theorem stating the relationship between profit, overall loss, and theft loss -/
theorem theft_loss_calculation (cost : ℝ) (cost_positive : cost > 0) :
  let selling_price := cost * (1 + profit_percentage / 100)
  let actual_revenue := cost * (1 - overall_loss_percentage / 100)
  selling_price * (1 - theft_loss_percentage / 100) = actual_revenue :=
sorry

end theft_loss_calculation_l3651_365179


namespace cube_edge_ratio_l3651_365148

theorem cube_edge_ratio (v₁ v₂ v₃ v₄ : ℝ) (h : v₁ / v₂ = 216 / 64 ∧ v₂ / v₃ = 64 / 27 ∧ v₃ / v₄ = 27 / 1) :
  ∃ (e₁ e₂ e₃ e₄ : ℝ), v₁ = e₁^3 ∧ v₂ = e₂^3 ∧ v₃ = e₃^3 ∧ v₄ = e₄^3 ∧ 
  e₁ / e₂ = 6 / 4 ∧ e₂ / e₃ = 4 / 3 ∧ e₃ / e₄ = 3 / 1 :=
by sorry

end cube_edge_ratio_l3651_365148


namespace elvis_squares_l3651_365101

theorem elvis_squares (total_matchsticks : ℕ) (elvis_square_size : ℕ) (ralph_square_size : ℕ) 
  (ralph_squares : ℕ) (leftover_matchsticks : ℕ) :
  total_matchsticks = 50 →
  elvis_square_size = 4 →
  ralph_square_size = 8 →
  ralph_squares = 3 →
  leftover_matchsticks = 6 →
  ∃ (elvis_squares : ℕ), 
    elvis_squares * elvis_square_size + ralph_squares * ralph_square_size + leftover_matchsticks = total_matchsticks ∧
    elvis_squares = 5 := by
  sorry

end elvis_squares_l3651_365101


namespace smallest_perimeter_of_rectangle_l3651_365127

theorem smallest_perimeter_of_rectangle (a b : ℕ) : 
  a * b = 1000 → 
  2 * (a + b) ≥ 130 ∧ 
  ∃ (x y : ℕ), x * y = 1000 ∧ 2 * (x + y) = 130 :=
by sorry

end smallest_perimeter_of_rectangle_l3651_365127


namespace function_inequality_l3651_365145

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x ∈ (Set.Ioo 0 (π / 2)), deriv f x * sin x < f x * cos x) →
  Real.sqrt 3 * f (π / 4) > Real.sqrt 2 * f (π / 3) := by
  sorry

end function_inequality_l3651_365145


namespace quadratic_equation_transform_sum_l3651_365160

theorem quadratic_equation_transform_sum (x r s : ℝ) : 
  (16 * x^2 - 64 * x - 144 = 0) →
  ((x + r)^2 = s) →
  (r + s = -7) :=
by sorry

end quadratic_equation_transform_sum_l3651_365160


namespace solution_set_of_inequality_l3651_365146

theorem solution_set_of_inequality (x : ℝ) :
  (((3 * x + 1) / (1 - 2 * x) ≥ 0) ↔ (-1/3 ≤ x ∧ x < 1/2)) :=
by sorry

end solution_set_of_inequality_l3651_365146


namespace solution_set_implies_sum_l3651_365170

theorem solution_set_implies_sum (a b : ℝ) : 
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) → a + b = -1 := by
  sorry

end solution_set_implies_sum_l3651_365170


namespace find_number_l3651_365119

theorem find_number (A B : ℕ+) : 
  Nat.gcd A B = 14 →
  Nat.lcm A B = 312 →
  B = 182 →
  A = 24 := by
sorry

end find_number_l3651_365119


namespace banana_arrangements_l3651_365196

/-- The number of distinct arrangements of letters in a word with repeated letters -/
def distinctArrangements (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The word "banana" has 6 letters total -/
def bananaLength : ℕ := 6

/-- The repeated letters in "banana" are [3, 2, 1] (for 'a', 'n', 'b' respectively) -/
def bananaRepeatedLetters : List ℕ := [3, 2, 1]

theorem banana_arrangements :
  distinctArrangements bananaLength bananaRepeatedLetters = 60 := by
  sorry

end banana_arrangements_l3651_365196


namespace area_is_192_l3651_365195

/-- A right triangle with a circle tangent to its legs -/
structure RightTriangleWithTangentCircle where
  /-- The circle cuts the hypotenuse into segments of lengths 1, 24, and 3 -/
  hypotenuse_segments : ℝ × ℝ × ℝ
  /-- The middle segment (of length 24) is a chord of the circle -/
  middle_segment_is_chord : hypotenuse_segments.2.1 = 24

/-- The area of a right triangle with a tangent circle satisfying specific conditions -/
def area (t : RightTriangleWithTangentCircle) : ℝ := sorry

/-- Theorem: The area of the triangle is 192 -/
theorem area_is_192 (t : RightTriangleWithTangentCircle) 
  (h1 : t.hypotenuse_segments.1 = 1)
  (h2 : t.hypotenuse_segments.2.2 = 3) : 
  area t = 192 := by sorry

end area_is_192_l3651_365195


namespace minimum_value_implies_a_l3651_365199

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

-- State the theorem
theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 0, f a x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 0, f a x = 1) →
  a = 3 :=
sorry

end minimum_value_implies_a_l3651_365199


namespace second_floor_cost_l3651_365130

/-- Represents the cost of rooms on each floor of an apartment building --/
structure ApartmentCosts where
  first_floor : ℕ
  second_floor : ℕ
  third_floor : ℕ

/-- Calculates the total monthly income from all rooms --/
def total_income (costs : ApartmentCosts) : ℕ :=
  3 * (costs.first_floor + costs.second_floor + costs.third_floor)

/-- Theorem stating the cost of rooms on the second floor --/
theorem second_floor_cost (costs : ApartmentCosts) :
  costs.first_floor = 15 →
  costs.third_floor = 2 * costs.first_floor →
  total_income costs = 165 →
  costs.second_floor = 10 := by
  sorry

#check second_floor_cost

end second_floor_cost_l3651_365130


namespace inequality_proof_l3651_365120

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := by
  sorry

end inequality_proof_l3651_365120


namespace triangle_ABC_properties_l3651_365141

/-- Given a triangle ABC with vertices A(4, 4), B(-4, 2), and C(2, 0) -/
def triangle_ABC : Set (ℝ × ℝ) := {(4, 4), (-4, 2), (2, 0)}

/-- The equation of a line ax + by + c = 0 is represented by the triple (a, b, c) -/
def Line := ℝ × ℝ × ℝ

/-- The median CD of triangle ABC -/
def median_CD : Line := sorry

/-- The altitude from C to AB -/
def altitude_C : Line := sorry

/-- The centroid G of triangle ABC -/
def centroid_G : ℝ × ℝ := sorry

theorem triangle_ABC_properties :
  (median_CD = (3, 2, -6)) ∧
  (altitude_C = (4, 1, -8)) ∧
  (centroid_G = (2/3, 2)) := by sorry

end triangle_ABC_properties_l3651_365141


namespace deal_or_no_deal_probability_l3651_365143

theorem deal_or_no_deal_probability (total : Nat) (desired : Nat) (chosen : Nat) 
  (h1 : total = 26)
  (h2 : desired = 9)
  (h3 : chosen = 1) :
  ∃ (removed : Nat), 
    (1 : ℚ) * desired / (total - removed - chosen) ≥ (1 : ℚ) / 2 ∧ 
    ∀ (r : Nat), r < removed → (1 : ℚ) * desired / (total - r - chosen) < (1 : ℚ) / 2 :=
by sorry

end deal_or_no_deal_probability_l3651_365143


namespace max_total_pieces_l3651_365109

/-- Represents a chessboard configuration -/
structure ChessboardConfig where
  white_pieces : ℕ
  black_pieces : ℕ

/-- The size of the chessboard -/
def board_size : ℕ := 8

/-- Condition: In each row and column, the number of white pieces is twice the number of black pieces -/
def valid_distribution (config : ChessboardConfig) : Prop :=
  config.white_pieces = 2 * config.black_pieces

/-- The total number of pieces on the board -/
def total_pieces (config : ChessboardConfig) : ℕ :=
  config.white_pieces + config.black_pieces

/-- The maximum number of pieces that can be placed on the board -/
def max_pieces : ℕ := board_size * board_size

theorem max_total_pieces :
  ∃ (config : ChessboardConfig),
    valid_distribution config ∧
    (∀ (other : ChessboardConfig),
      valid_distribution other →
      total_pieces other ≤ total_pieces config) ∧
    total_pieces config = 48 :=
  sorry

end max_total_pieces_l3651_365109


namespace factorization_problem_l3651_365190

theorem factorization_problem (a m n b : ℝ) : 
  (∀ x, x^2 + a*x + m = (x + 2) * (x + 4)) →
  (∀ x, x^2 + n*x + b = (x + 1) * (x + 9)) →
  (∀ x, x^2 + a*x + b = (x + 3)^2) :=
by sorry

end factorization_problem_l3651_365190


namespace fraction_equivalence_l3651_365171

theorem fraction_equivalence (a b : ℚ) : 
  (a ≠ 0) → (b ≠ 0) → ((1 / (a / b)) * (5 / 6) = 1 / (5 / 2)) → (a / b = 25 / 12) := by
  sorry

end fraction_equivalence_l3651_365171


namespace fraction_calculation_l3651_365181

theorem fraction_calculation : 
  (((1 : ℚ) / 2 + (1 : ℚ) / 5) / ((3 : ℚ) / 7 - (1 : ℚ) / 14)) * (2 : ℚ) / 3 = 98 / 75 := by
  sorry

end fraction_calculation_l3651_365181


namespace orange_bin_calculation_l3651_365194

/-- Calculates the final number of oranges in a bin after a series of transactions -/
theorem orange_bin_calculation (initial : ℕ) (sold : ℕ) (new_shipment : ℕ) : 
  initial = 124 → sold = 46 → new_shipment = 250 → 
  (initial - sold - (initial - sold) / 2 + new_shipment) = 289 := by
  sorry

end orange_bin_calculation_l3651_365194


namespace sam_and_joan_books_l3651_365152

/-- Given that Sam has 110 books and Joan has 102 books, prove that they have 212 books together. -/
theorem sam_and_joan_books : 
  let sam_books : ℕ := 110
  let joan_books : ℕ := 102
  sam_books + joan_books = 212 :=
by sorry

end sam_and_joan_books_l3651_365152


namespace selling_price_calculation_l3651_365185

def cost_price : ℝ := 225
def profit_percentage : ℝ := 20

theorem selling_price_calculation :
  let profit := (profit_percentage / 100) * cost_price
  let selling_price := cost_price + profit
  selling_price = 270 := by
sorry

end selling_price_calculation_l3651_365185


namespace unique_arithmetic_progression_l3651_365125

theorem unique_arithmetic_progression : ∃! (a b : ℝ),
  (a - 15 = b - a) ∧ (ab - b = b - a) ∧ (a - b = 5) ∧ (a = 10) ∧ (b = 5) := by
  sorry

end unique_arithmetic_progression_l3651_365125


namespace exists_valid_marking_configuration_l3651_365159

/-- A type representing a cell in the grid -/
structure Cell :=
  (row : Fin 19)
  (col : Fin 19)

/-- A type representing a marking configuration of the grid -/
def MarkingConfiguration := Cell → Bool

/-- A function to count marked cells in a 10x10 square -/
def countMarkedCells (config : MarkingConfiguration) (topLeft : Cell) : Nat :=
  sorry

/-- A predicate to check if all 10x10 squares have different counts -/
def allSquaresDifferent (config : MarkingConfiguration) : Prop :=
  ∀ s1 s2 : Cell, s1 ≠ s2 → 
    countMarkedCells config s1 ≠ countMarkedCells config s2

/-- The main theorem stating the existence of a valid marking configuration -/
theorem exists_valid_marking_configuration : 
  ∃ (config : MarkingConfiguration), allSquaresDifferent config :=
sorry

end exists_valid_marking_configuration_l3651_365159


namespace defective_components_probability_l3651_365176

-- Define the probability function
def probability (p q r : ℕ) : ℚ :=
  let total_components := p + q
  let numerator := q * (Nat.descFactorial (r-1) (q-1)) * (Nat.descFactorial p (r-q)) +
                   p * (Nat.descFactorial (r-1) (p-1)) * (Nat.descFactorial q (r-p))
  let denominator := Nat.descFactorial total_components r
  ↑numerator / ↑denominator

-- State the theorem
theorem defective_components_probability (p q r : ℕ) 
  (h1 : q < p) (h2 : p < r) (h3 : r < p + q) :
  probability p q r = (↑q * Nat.descFactorial (r-1) (q-1) * Nat.descFactorial p (r-q) + 
                       ↑p * Nat.descFactorial (r-1) (p-1) * Nat.descFactorial q (r-p)) / 
                      Nat.descFactorial (p+q) r :=
by
  sorry


end defective_components_probability_l3651_365176


namespace triangle_semiperimeter_from_side_and_excircle_radii_l3651_365128

/-- Given a side 'a' of a triangle and the radii of the excircles opposite 
    the other two sides 'ρ_b' and 'ρ_c', the semiperimeter 's' of the 
    triangle is equal to a/2 + √((a/2)² + ρ_b * ρ_c). -/
theorem triangle_semiperimeter_from_side_and_excircle_radii 
  (a ρ_b ρ_c : ℝ) (ha : a > 0) (hb : ρ_b > 0) (hc : ρ_c > 0) :
  ∃ s : ℝ, s > 0 ∧ s = a / 2 + Real.sqrt ((a / 2)^2 + ρ_b * ρ_c) := by
  sorry

end triangle_semiperimeter_from_side_and_excircle_radii_l3651_365128


namespace house_painting_theorem_l3651_365149

/-- Represents the number of worker-hours required to paint a house -/
def totalWorkerHours : ℕ := 32

/-- Represents the number of people who started painting -/
def initialWorkers : ℕ := 6

/-- Represents the number of hours the initial workers painted -/
def initialHours : ℕ := 2

/-- Represents the total time available to paint the house -/
def totalTime : ℕ := 4

/-- Calculates the number of additional workers needed to complete the painting -/
def additionalWorkersNeeded : ℕ :=
  (totalWorkerHours - initialWorkers * initialHours) / (totalTime - initialHours) - initialWorkers

theorem house_painting_theorem :
  additionalWorkersNeeded = 4 := by
  sorry

#eval additionalWorkersNeeded

end house_painting_theorem_l3651_365149


namespace percentage_girls_like_basketball_l3651_365166

/-- Given a class with the following properties:
  * There are 25 students in total
  * 60% of students are girls
  * 40% of boys like playing basketball
  * The number of girls who like basketball is double the number of boys who don't like it
  Prove that 80% of girls like playing basketball -/
theorem percentage_girls_like_basketball :
  ∀ (total_students : ℕ) 
    (girls boys boys_like_basketball boys_dont_like_basketball girls_like_basketball : ℕ),
  total_students = 25 →
  girls = (60 : ℕ) * total_students / 100 →
  boys = total_students - girls →
  boys_like_basketball = (40 : ℕ) * boys / 100 →
  boys_dont_like_basketball = boys - boys_like_basketball →
  girls_like_basketball = 2 * boys_dont_like_basketball →
  (girls_like_basketball : ℚ) / girls * 100 = 80 := by
sorry

end percentage_girls_like_basketball_l3651_365166


namespace joan_has_ten_books_l3651_365162

/-- The number of books Tom has -/
def tom_books : ℕ := 38

/-- The total number of books Joan and Tom have together -/
def total_books : ℕ := 48

/-- The number of books Joan has -/
def joan_books : ℕ := total_books - tom_books

theorem joan_has_ten_books : joan_books = 10 := by
  sorry

end joan_has_ten_books_l3651_365162
