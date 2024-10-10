import Mathlib

namespace midpoint_trajectory_l967_96726

/-- The trajectory of the midpoint of a perpendicular from a point on the unit circle to the x-axis -/
theorem midpoint_trajectory (a b x y : ℝ) : 
  a^2 + b^2 = 1 →  -- P(a, b) is on the unit circle
  x = a →          -- x-coordinate of M is same as P
  y = b / 2 →      -- y-coordinate of M is half of P's
  x^2 + 4 * y^2 = 1 := by
sorry

end midpoint_trajectory_l967_96726


namespace ln_abs_properties_l967_96776

-- Define the function f(x) = ln|x|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

-- State the theorem
theorem ln_abs_properties :
  (∀ x ≠ 0, f (-x) = f x) ∧  -- f is even
  (∀ x y, 0 < x → x < y → f x < f y) :=  -- f is increasing on (0, +∞)
by sorry

end ln_abs_properties_l967_96776


namespace ellipse_properties_l967_96743

/-- Ellipse C passing through points A(2,0) and B(0,1) -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Line on which point P lies -/
def line_P (x y : ℝ) : Prop := x + y = 4

/-- Point Q on ellipse C -/
def point_Q (x y : ℝ) : Prop := ellipse_C x y

/-- Parallelogram condition for PAQB -/
def is_parallelogram (px py qx qy : ℝ) : Prop :=
  px + qx = 2 ∧ py + qy = 1

theorem ellipse_properties :
  ∃ (e : ℝ),
    (∀ x y, ellipse_C x y ↔ x^2 / 4 + y^2 = 1) ∧
    e = Real.sqrt 3 / 2 ∧
    ∃ px py qx qy,
      line_P px py ∧
      point_Q qx qy ∧
      is_parallelogram px py qx qy ∧
      ((px = 18/5 ∧ py = 2/5) ∨ (px = 2 ∧ py = 2)) :=
by sorry

end ellipse_properties_l967_96743


namespace arithmetic_sequence_fifth_term_l967_96753

/-- Given an arithmetic sequence with the first four terms as specified,
    prove that the fifth term has the given form. -/
theorem arithmetic_sequence_fifth_term
  (x y : ℝ)
  (seq : ℕ → ℝ)
  (h1 : seq 0 = x + y^2)
  (h2 : seq 1 = x + 2*y)
  (h3 : seq 2 = x*y^2)
  (h4 : seq 3 = x/y^2)
  (h_arithmetic : ∀ n : ℕ, seq (n + 1) - seq n = seq 1 - seq 0) :
  seq 4 = (y^6 - 2*y^5 + 4*y) / (y^4 + y^2) :=
sorry

end arithmetic_sequence_fifth_term_l967_96753


namespace unique_fraction_decomposition_l967_96761

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (m n : ℕ), m ≠ n ∧ m > 0 ∧ n > 0 ∧ (2 : ℚ) / p = 1 / n + 1 / m ∧
  n = (p + 1) / 2 ∧ m = p * (p + 1) / 2 := by
  sorry

end unique_fraction_decomposition_l967_96761


namespace average_weight_proof_l967_96739

theorem average_weight_proof (rachel_weight jimmy_weight adam_weight : ℝ) : 
  rachel_weight = 75 ∧ 
  rachel_weight = jimmy_weight - 6 ∧ 
  rachel_weight = adam_weight + 15 → 
  (rachel_weight + jimmy_weight + adam_weight) / 3 = 72 := by
sorry

end average_weight_proof_l967_96739


namespace sum_of_roots_equals_three_l967_96730

theorem sum_of_roots_equals_three : ∃ (P Q : ℝ), P + Q = 3 ∧ 3 * P^2 - 9 * P + 6 = 0 ∧ 3 * Q^2 - 9 * Q + 6 = 0 := by
  sorry

end sum_of_roots_equals_three_l967_96730


namespace valid_arrangements_l967_96796

/-- Represents the number of ways to arrange 7 people in a line -/
def arrangement_count : ℕ := 72

/-- Represents the total number of people -/
def total_people : ℕ := 7

/-- Represents the number of students -/
def student_count : ℕ := 6

/-- Represents whether two people are at the ends of the line -/
def are_at_ends (coach : ℕ) (student_a : ℕ) : Prop :=
  (coach = 1 ∧ student_a = total_people) ∨ (coach = total_people ∧ student_a = 1)

/-- Represents whether two students are adjacent in the line -/
def are_adjacent (student1 : ℕ) (student2 : ℕ) : Prop :=
  student1 + 1 = student2 ∨ student2 + 1 = student1

/-- Represents whether two students are not adjacent in the line -/
def are_not_adjacent (student1 : ℕ) (student2 : ℕ) : Prop :=
  ¬(are_adjacent student1 student2)

/-- Theorem stating that the number of valid arrangements is 72 -/
theorem valid_arrangements :
  ∀ (coach student_a student_b student_c student_d : ℕ),
    are_at_ends coach student_a →
    are_adjacent student_b student_c →
    are_not_adjacent student_b student_d →
    arrangement_count = 72 := by sorry

end valid_arrangements_l967_96796


namespace f_monotone_increasing_interval_l967_96709

-- Define the function f(x) = x^2 + 2x + 3
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem about the monotonically increasing interval of f
theorem f_monotone_increasing_interval :
  ∃ (a : ℝ), a = -1 ∧
  ∀ (x y : ℝ), x > a → y > x → f y > f x :=
sorry

end f_monotone_increasing_interval_l967_96709


namespace no_solution_condition_l967_96701

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (1 / (x - 2) + a / (2 - x) ≠ 2 * a)) ↔ (a = 0 ∨ a = 1) := by
  sorry

end no_solution_condition_l967_96701


namespace total_tickets_sold_l967_96706

/-- Proves that the total number of tickets sold is 350 --/
theorem total_tickets_sold (orchestra_price balcony_price : ℕ)
  (total_cost : ℕ) (balcony_excess : ℕ) :
  orchestra_price = 12 →
  balcony_price = 8 →
  total_cost = 3320 →
  balcony_excess = 90 →
  ∃ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets * orchestra_price + balcony_tickets * balcony_price = total_cost ∧
    balcony_tickets = orchestra_tickets + balcony_excess ∧
    orchestra_tickets + balcony_tickets = 350 :=
by sorry

end total_tickets_sold_l967_96706


namespace system_solution_l967_96777

theorem system_solution (x y z b : ℝ) : 
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧
  (x + y + z = 2 * b) ∧
  (x^2 + y^2 + z^2 = b^2) →
  ((x = 0 ∧ y = -z) ∨ (y = 0 ∧ x = -z)) ∧ b = 0 :=
by sorry

end system_solution_l967_96777


namespace only_solutions_for_equation_l967_96756

theorem only_solutions_for_equation (x y : ℕ) : 
  33^x + 31 = 2^y ↔ (x = 0 ∧ y = 5) ∨ (x = 1 ∧ y = 6) := by
  sorry

end only_solutions_for_equation_l967_96756


namespace marks_speed_l967_96714

/-- Given a distance of 24 miles and a time of 4 hours, the speed is 6 miles per hour. -/
theorem marks_speed (distance : ℝ) (time : ℝ) (h1 : distance = 24) (h2 : time = 4) :
  distance / time = 6 := by
  sorry

end marks_speed_l967_96714


namespace star_seven_three_l967_96759

-- Define the ⋆ operation
def star (a b : ℤ) : ℤ := 4*a + 3*b - a*b

-- State the theorem
theorem star_seven_three : star 7 3 = 16 := by
  sorry

end star_seven_three_l967_96759


namespace root_in_interval_l967_96729

theorem root_in_interval (a : ℤ) : 
  (∃ x : ℝ, x > a ∧ x < a + 1 ∧ Real.log x + x - 4 = 0) → a = 2 := by
  sorry

end root_in_interval_l967_96729


namespace expression_simplification_l967_96765

theorem expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * (a^2)^(1/3) * b^(1/2) * (-6 * a^(1/3) * b^(1/3))^2) / (-3 * (a*b^5)^(1/6)) = -24 * a^(7/6) * b^(1/3) := by
  sorry

end expression_simplification_l967_96765


namespace quadratic_real_solutions_l967_96713

-- Define the quadratic equations
def quadratic1 (a x : ℝ) : ℝ := a * x^2 + (a + 1) * x - 2
def quadratic2 (a x : ℝ) : ℝ := (1 - a) * x^2 + (a + 1) * x - 2

-- Define the conditions for real solutions
def realSolutions1 (a : ℝ) : Prop :=
  a < -5 - 2 * Real.sqrt 6 ∨ (2 * Real.sqrt 6 - 5 < a ∧ a < 0) ∨ a > 0

def realSolutions2 (a : ℝ) : Prop :=
  a < 1 ∨ (1 < a ∧ a < 3) ∨ a > 3

-- Theorem statement
theorem quadratic_real_solutions :
  ∀ a : ℝ,
    (∃ x : ℝ, quadratic1 a x = 0) ↔ realSolutions1 a ∧
    (∃ x : ℝ, quadratic2 a x = 0) ↔ realSolutions2 a :=
by sorry

end quadratic_real_solutions_l967_96713


namespace alexander_pencils_per_picture_l967_96773

theorem alexander_pencils_per_picture
  (first_exhibition_pictures : ℕ)
  (new_galleries : ℕ)
  (pictures_per_new_gallery : ℕ)
  (signing_pencils_per_exhibition : ℕ)
  (total_pencils_used : ℕ)
  (h1 : first_exhibition_pictures = 9)
  (h2 : new_galleries = 5)
  (h3 : pictures_per_new_gallery = 2)
  (h4 : signing_pencils_per_exhibition = 2)
  (h5 : total_pencils_used = 88) :
  (total_pencils_used - (signing_pencils_per_exhibition * (new_galleries + 1))) /
  (first_exhibition_pictures + new_galleries * pictures_per_new_gallery) = 4 := by
sorry

end alexander_pencils_per_picture_l967_96773


namespace intersection_of_lines_l967_96700

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℚ
  y : ℚ

/-- Defines a line in 2D space using the equation y = mx + b -/
structure Line where
  m : ℚ  -- slope
  b : ℚ  -- y-intercept

def line1 : Line := { m := 3, b := 0 }
def line2 : Line := { m := -7, b := 5 }

theorem intersection_of_lines (l1 l2 : Line) : 
  ∃! p : IntersectionPoint, 
    p.y = l1.m * p.x + l1.b ∧ 
    p.y = l2.m * p.x + l2.b := by
  sorry

#check intersection_of_lines line1 line2

end intersection_of_lines_l967_96700


namespace smallest_multiples_sum_l967_96779

theorem smallest_multiples_sum : ∃ c d : ℕ,
  (c ≥ 10 ∧ c < 100 ∧ c % 5 = 0 ∧ ∀ x : ℕ, x ≥ 10 ∧ x < 100 ∧ x % 5 = 0 → c ≤ x) ∧
  (d ≥ 100 ∧ d < 1000 ∧ d % 7 = 0 ∧ ∀ y : ℕ, y ≥ 100 ∧ y < 1000 ∧ y % 7 = 0 → d ≤ y) ∧
  c + d = 115 :=
by sorry

end smallest_multiples_sum_l967_96779


namespace circle_center_and_radius_l967_96797

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2 -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_center_and_radius :
  ∃ (h k r : ℝ), 
    (∀ x y : ℝ, CircleEquation h k r x y ↔ (x - 2)^2 + (y + 3)^2 = 2) ∧
    h = 2 ∧ k = -3 ∧ r = Real.sqrt 2 := by
  sorry

end circle_center_and_radius_l967_96797


namespace total_coins_l967_96778

/-- Given a number of stacks and coins per stack, proves that the total number of coins
    is equal to the product of these two quantities. -/
theorem total_coins (num_stacks : ℕ) (coins_per_stack : ℕ) :
  num_stacks * coins_per_stack = num_stacks * coins_per_stack := by
  sorry

/-- Calculates the total number of coins Maria has. -/
def maria_coins : ℕ :=
  let num_stacks : ℕ := 10
  let coins_per_stack : ℕ := 6
  num_stacks * coins_per_stack

#eval maria_coins

end total_coins_l967_96778


namespace total_oil_needed_l967_96769

/-- Represents the oil requirements for a bicycle -/
structure BikeOil where
  wheel : ℕ  -- Oil needed for one wheel
  chain : ℕ  -- Oil needed for the chain
  pedals : ℕ -- Oil needed for the pedals
  brakes : ℕ -- Oil needed for the brakes

/-- Calculates the total oil needed for a bicycle -/
def totalOilForBike (bike : BikeOil) : ℕ :=
  2 * bike.wheel + bike.chain + bike.pedals + bike.brakes

/-- The oil requirements for the first bicycle -/
def bike1 : BikeOil := 
  { wheel := 20, chain := 15, pedals := 8, brakes := 10 }

/-- The oil requirements for the second bicycle -/
def bike2 : BikeOil := 
  { wheel := 25, chain := 18, pedals := 10, brakes := 12 }

/-- The oil requirements for the third bicycle -/
def bike3 : BikeOil := 
  { wheel := 30, chain := 20, pedals := 12, brakes := 15 }

/-- Theorem stating the total oil needed for all three bicycles -/
theorem total_oil_needed : 
  totalOilForBike bike1 + totalOilForBike bike2 + totalOilForBike bike3 = 270 := by
  sorry

end total_oil_needed_l967_96769


namespace incorrect_value_calculation_l967_96749

theorem incorrect_value_calculation (n : ℕ) (initial_mean correct_mean correct_value : ℝ) 
  (h1 : n = 25)
  (h2 : initial_mean = 190)
  (h3 : correct_mean = 191.4)
  (h4 : correct_value = 165) :
  ∃ incorrect_value : ℝ,
    incorrect_value = n * correct_mean - (n - 1) * initial_mean - correct_value ∧
    incorrect_value = 200 := by
  sorry

end incorrect_value_calculation_l967_96749


namespace water_formed_hcl_nahco3_l967_96771

/-- Represents a chemical compound in a reaction -/
structure Compound where
  name : String
  moles : ℚ

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Compound
  products : List Compound

/-- The balanced chemical equation for the reaction of HCl and NaHCO₃ -/
def hcl_nahco3_reaction : Reaction :=
  { reactants := [
      { name := "HCl", moles := 1 },
      { name := "NaHCO₃", moles := 1 }
    ],
    products := [
      { name := "NaCl", moles := 1 },
      { name := "CO₂", moles := 1 },
      { name := "H₂O", moles := 1 }
    ]
  }

/-- Calculate the amount of a specific product formed in a reaction -/
def amount_formed (reaction : Reaction) (product_name : String) (limiting_reagent_moles : ℚ) : ℚ :=
  let product := reaction.products.find? (fun c => c.name = product_name)
  match product with
  | some p => p.moles * limiting_reagent_moles
  | none => 0

/-- Theorem: The amount of water formed when 2 moles of HCl react with 2 moles of NaHCO₃ is 2 moles -/
theorem water_formed_hcl_nahco3 :
  amount_formed hcl_nahco3_reaction "H₂O" 2 = 2 := by
  sorry

end water_formed_hcl_nahco3_l967_96771


namespace f_image_is_zero_to_eight_l967_96792

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the domain
def D : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- Theorem statement
theorem f_image_is_zero_to_eight :
  Set.image f D = { y | 0 ≤ y ∧ y ≤ 8 } := by
  sorry

end f_image_is_zero_to_eight_l967_96792


namespace no_odd_integer_solution_l967_96707

theorem no_odd_integer_solution (n : ℕ+) (x y z : ℤ) 
  (hx : Odd x) (hy : Odd y) (hz : Odd z) : 
  (x + y)^n.val + (y + z)^n.val ≠ (x + z)^n.val := by
  sorry

end no_odd_integer_solution_l967_96707


namespace min_value_of_expression_l967_96788

theorem min_value_of_expression (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  let A : ℝ × ℝ := (-4, 0)
  let B : ℝ × ℝ := (-1, 0)
  let P : ℝ × ℝ := (a, b)
  (‖P - A‖ = 2 * ‖P - B‖) →
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 
    let Q : ℝ × ℝ := (x, y)
    (‖Q - A‖ = 2 * ‖Q - B‖) → 
    (4 / a^2 + 1 / b^2 ≤ 4 / x^2 + 1 / y^2)) →
  4 / a^2 + 1 / b^2 = 9/4 :=
by sorry

end min_value_of_expression_l967_96788


namespace popsicle_sticks_per_boy_l967_96757

theorem popsicle_sticks_per_boy (num_boys num_girls : ℕ) (sticks_per_girl : ℕ) (diff : ℕ) :
  num_boys = 10 →
  num_girls = 12 →
  sticks_per_girl = 12 →
  num_girls * sticks_per_girl + diff = num_boys * (num_girls * sticks_per_girl + diff) / num_boys →
  diff = 6 →
  (num_girls * sticks_per_girl + diff) / num_boys = 15 :=
by sorry

end popsicle_sticks_per_boy_l967_96757


namespace total_distance_walked_l967_96794

/-- The distance Spencer walked from his house to the library -/
def distance_house_to_library : ℝ := 0.3

/-- The distance Spencer walked from the library to the post office -/
def distance_library_to_post_office : ℝ := 0.1

/-- The distance Spencer walked from the post office back home -/
def distance_post_office_to_house : ℝ := 0.4

/-- The theorem stating that the total distance Spencer walked is 0.8 miles -/
theorem total_distance_walked :
  distance_house_to_library + distance_library_to_post_office + distance_post_office_to_house = 0.8 := by
  sorry

end total_distance_walked_l967_96794


namespace range_of_n_l967_96732

theorem range_of_n (m n : ℝ) : (m^2 - 2*m)^2 + 4*m^2 - 8*m + 6 - n = 0 → n ≥ 3 := by
  sorry

end range_of_n_l967_96732


namespace max_value_rational_function_max_value_attained_l967_96795

theorem max_value_rational_function (x : ℝ) :
  x^6 / (x^12 + 3*x^9 - 5*x^6 + 15*x^3 + 27) ≤ 1/37 :=
by sorry

theorem max_value_attained :
  ∃ x : ℝ, x^6 / (x^12 + 3*x^9 - 5*x^6 + 15*x^3 + 27) = 1/37 :=
by sorry

end max_value_rational_function_max_value_attained_l967_96795


namespace good_number_proof_l967_96747

theorem good_number_proof :
  ∃! n : ℕ, n ∈ Finset.range 2016 ∧
  (Finset.sum (Finset.range 2016) id - n) % 2016 = 0 ∧
  n = 1008 := by
sorry

end good_number_proof_l967_96747


namespace double_reflection_result_l967_96708

/-- Reflect a point about the line y=x -/
def reflectYEqualsX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Reflect a point about the line y=-x -/
def reflectYEqualsNegX (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- The final position after two reflections -/
def finalPosition (p : ℝ × ℝ) : ℝ × ℝ :=
  reflectYEqualsNegX (reflectYEqualsX p)

theorem double_reflection_result :
  finalPosition (3, -7) = (3, 7) := by
  sorry

end double_reflection_result_l967_96708


namespace triangle_area_345_l967_96764

theorem triangle_area_345 (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  (1/2 : ℝ) * a * b = 6 := by
  sorry

end triangle_area_345_l967_96764


namespace function_relationship_l967_96752

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y)
variable (h2 : ∀ x, f (x + 2) = f (-x + 2))

-- State the theorem
theorem function_relationship : f 2.5 > f 1 ∧ f 1 > f 3.5 :=
sorry

end function_relationship_l967_96752


namespace complex_number_classification_real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l967_96728

def complex_number (m : ℝ) : ℂ := (1 + Complex.I) * m^2 - m * (5 + 3 * Complex.I) + 6

theorem complex_number_classification (m : ℝ) :
  (complex_number m).im = 0 ∨ 
  ((complex_number m).re ≠ 0 ∧ (complex_number m).im ≠ 0) ∨ 
  ((complex_number m).re = 0 ∧ (complex_number m).im ≠ 0) :=
by
  sorry

theorem real_number_condition (m : ℝ) :
  (complex_number m).im = 0 ↔ m = 0 ∨ m = 3 :=
by
  sorry

theorem imaginary_number_condition (m : ℝ) :
  ((complex_number m).re ≠ 0 ∧ (complex_number m).im ≠ 0) ↔ m ≠ 0 ∧ m ≠ 3 :=
by
  sorry

theorem pure_imaginary_number_condition (m : ℝ) :
  ((complex_number m).re = 0 ∧ (complex_number m).im ≠ 0) ↔ m = 2 :=
by
  sorry

end complex_number_classification_real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l967_96728


namespace arithmetic_sequence_ninth_term_l967_96731

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the third term is 20 and the sixth term is 26,
    prove that the ninth term is 32. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third_term : a 3 = 20)
  (h_sixth_term : a 6 = 26) :
  a 9 = 32 := by
  sorry


end arithmetic_sequence_ninth_term_l967_96731


namespace correct_percentage_calculation_l967_96791

/-- Calculates the overall percentage of correct answers across multiple tests -/
def overallPercentage (testSizes : List Nat) (scores : List Rat) : Rat :=
  sorry

/-- Rounds a rational number to the nearest whole number -/
def roundToNearest (x : Rat) : Nat :=
  sorry

theorem correct_percentage_calculation :
  let testSizes : List Nat := [40, 30, 20]
  let scores : List Rat := [65/100, 85/100, 75/100]
  roundToNearest (overallPercentage testSizes scores * 100) = 74 :=
by sorry

end correct_percentage_calculation_l967_96791


namespace alpha_beta_inequality_l967_96737

theorem alpha_beta_inequality (α β : ℝ) : α > β ↔ α - β > Real.sin α - Real.sin β := by
  sorry

end alpha_beta_inequality_l967_96737


namespace pants_cost_is_correct_l967_96705

/-- The cost of pants given total payment, cost of shirt, and change received -/
def cost_of_pants (total_payment shirt_cost change : ℚ) : ℚ :=
  total_payment - shirt_cost - change

/-- Theorem stating the cost of pants is $9.24 given the problem conditions -/
theorem pants_cost_is_correct :
  let total_payment : ℚ := 20
  let shirt_cost : ℚ := 8.25
  let change : ℚ := 2.51
  cost_of_pants total_payment shirt_cost change = 9.24 := by
  sorry

end pants_cost_is_correct_l967_96705


namespace inequality_proof_l967_96702

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : c < 1) : (a - b) * (c - 1) < 0 := by
  sorry

end inequality_proof_l967_96702


namespace power_of_ten_plus_one_divisibility_not_always_divisible_by_nine_l967_96720

theorem power_of_ten_plus_one_divisibility (n : ℕ) :
  (9 ∣ 10^n + 1) → (9 ∣ 10^(n+1) + 1) :=
by sorry

theorem not_always_divisible_by_nine :
  ∃ n : ℕ, ¬(9 ∣ 10^n + 1) :=
by sorry

end power_of_ten_plus_one_divisibility_not_always_divisible_by_nine_l967_96720


namespace inequality_proof_l967_96754

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a + b + c = 1) : 
  (a^4 + b^4)/(a^6 + b^6) + (b^4 + c^4)/(b^6 + c^6) + (c^4 + a^4)/(c^6 + a^6) ≤ 1/(a*b*c) := by
  sorry

#check inequality_proof

end inequality_proof_l967_96754


namespace inequality_solution_l967_96784

def solution_set : Set ℝ := {x : ℝ | -5 < x ∧ x < 1 ∨ x > 6}

theorem inequality_solution :
  {x : ℝ | (x - 1) / (x^2 - x - 30) > 0} = solution_set :=
by sorry

end inequality_solution_l967_96784


namespace ceiling_negative_three_point_seven_l967_96703

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end ceiling_negative_three_point_seven_l967_96703


namespace inequality_proof_l967_96722

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (1 + a^2) + Real.sqrt (1 + b^2) ≥ Real.sqrt 5 := by
  sorry

end inequality_proof_l967_96722


namespace popcorn_probability_l967_96760

theorem popcorn_probability (total : ℝ) (h_total_pos : 0 < total) : 
  let white := (3/4 : ℝ) * total
  let yellow := (1/4 : ℝ) * total
  let white_popped := (3/5 : ℝ) * white
  let yellow_popped := (3/4 : ℝ) * yellow
  let total_popped := white_popped + yellow_popped
  (white_popped / total_popped) = (12/17 : ℝ) := by
sorry

end popcorn_probability_l967_96760


namespace equality_from_fraction_equality_l967_96704

theorem equality_from_fraction_equality (a b c d : ℝ) :
  (a + b) / (c + d) = (b + c) / (a + d) ∧ 
  (a + b) / (c + d) ≠ -1 →
  a = c :=
by sorry

end equality_from_fraction_equality_l967_96704


namespace trailing_zeroes_sum_factorials_l967_96785

def trailing_zeroes (n : ℕ) : ℕ := sorry

def factorial (n : ℕ) : ℕ := sorry

theorem trailing_zeroes_sum_factorials :
  trailing_zeroes (factorial 60 + factorial 120) = trailing_zeroes (factorial 60) :=
sorry

end trailing_zeroes_sum_factorials_l967_96785


namespace even_digits_in_512_base_8_l967_96745

/-- Represents a natural number in base 8 as a list of digits -/
def BaseEightRepresentation : Type := List Nat

/-- Converts a natural number to its base-8 representation -/
def toBaseEight (n : Nat) : BaseEightRepresentation :=
  sorry

/-- Counts the number of even digits in a base-8 representation -/
def countEvenDigits (rep : BaseEightRepresentation) : Nat :=
  sorry

theorem even_digits_in_512_base_8 :
  countEvenDigits (toBaseEight 512) = 3 := by
  sorry

end even_digits_in_512_base_8_l967_96745


namespace number_difference_l967_96768

theorem number_difference (a b : ℕ) (h1 : a + b = 72) (h2 : a = 30) (h3 : b = 42) :
  b - a = 12 := by
  sorry

end number_difference_l967_96768


namespace nathan_warmth_increase_l967_96780

def blankets_in_closet : ℕ := 14
def warmth_per_blanket : ℕ := 3

def warmth_increase (blankets_used : ℕ) : ℕ :=
  blankets_used * warmth_per_blanket

theorem nathan_warmth_increase :
  warmth_increase (blankets_in_closet / 2) = 21 := by
  sorry

end nathan_warmth_increase_l967_96780


namespace no_solution_iff_a_in_range_l967_96715

/-- The equation has no solutions if and only if a is in the specified range -/
theorem no_solution_iff_a_in_range (a : ℝ) : 
  (∀ x : ℝ, 5*|x - 4*a| + |x - a^2| + 4*x - 4*a ≠ 0) ↔ 
  (a < -8 ∨ a > 0) := by
sorry

end no_solution_iff_a_in_range_l967_96715


namespace dice_roll_probability_l967_96793

def is_valid_roll (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6

def meets_conditions (a b c : ℕ) : Prop :=
  a * b * c = 72 ∧ a + b + c = 13

def total_outcomes : ℕ := 6 * 6 * 6

def favorable_outcomes : ℕ := 6

theorem dice_roll_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 36 :=
sorry

end dice_roll_probability_l967_96793


namespace johnny_money_left_l967_96790

def savings_september : ℝ := 30
def savings_october : ℝ := 49
def savings_november : ℝ := 46
def savings_december : ℝ := 55
def january_savings_increase : ℝ := 0.15
def video_game_cost : ℝ := 58
def book_cost : ℝ := 25
def birthday_present_cost : ℝ := 40

def total_savings : ℝ :=
  savings_september + savings_october + savings_november + savings_december +
  (savings_december * (1 + january_savings_increase))

def total_expenses : ℝ :=
  video_game_cost + book_cost + birthday_present_cost

theorem johnny_money_left :
  total_savings - total_expenses = 120.25 := by
  sorry

end johnny_money_left_l967_96790


namespace solve_for_z_l967_96738

theorem solve_for_z : ∃ z : ℝ, (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt z = 2.4 → z = 75 := by
  sorry

end solve_for_z_l967_96738


namespace average_weight_increase_problem_solution_l967_96735

/-- The increase in average weight when replacing a person in a group -/
theorem average_weight_increase (n : ℕ) (old_weight new_weight : ℝ) : 
  n > 0 → (new_weight - old_weight) / n = (new_weight - old_weight) / n := by
  sorry

/-- The specific case of the problem -/
theorem problem_solution : 
  let n : ℕ := 10
  let old_weight : ℝ := 65
  let new_weight : ℝ := 137
  (new_weight - old_weight) / n = 7.2 := by
  sorry

end average_weight_increase_problem_solution_l967_96735


namespace units_digit_of_sum_64_75_base_8_l967_96789

/-- Represents a number in base 8 --/
def OctalNum := Nat

/-- Converts a base 10 number to its base 8 representation --/
def toOctal (n : Nat) : OctalNum := sorry

/-- Adds two numbers in base 8 --/
def octalAdd (a b : OctalNum) : OctalNum := sorry

/-- Gets the units digit of a number in base 8 --/
def unitsDigit (n : OctalNum) : Nat := sorry

theorem units_digit_of_sum_64_75_base_8 :
  unitsDigit (octalAdd (toOctal 64) (toOctal 75)) = 1 := by sorry

end units_digit_of_sum_64_75_base_8_l967_96789


namespace min_reciprocal_sum_l967_96781

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (1 / x + 1 / y) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end min_reciprocal_sum_l967_96781


namespace circumcircle_area_l967_96770

theorem circumcircle_area (a b c : ℝ) (A B C : ℝ) (R : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  c / Real.sin C = 2 * R ∧
  A = 5 * π / 12 ∧
  B = π / 4 ∧
  c = 4 →
  π * R^2 = 16 * π / 3 := by
sorry

end circumcircle_area_l967_96770


namespace nonzero_terms_count_l967_96712

-- Define the polynomials
def p (x : ℝ) := x^2 + 2
def q (x : ℝ) := 3*x^3 + 5*x^2 + 2
def r (x : ℝ) := x^4 - 3*x^3 + 2*x^2

-- Define the expression
def expression (x : ℝ) := p x * q x - 2 * r x

-- Theorem statement
theorem nonzero_terms_count : 
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
  ∀ x, expression x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e :=
sorry

end nonzero_terms_count_l967_96712


namespace green_face_probability_octahedral_die_l967_96782

/-- An octahedral die with green and yellow faces -/
structure OctahedralDie where
  total_faces : Nat
  green_faces : Nat
  yellow_faces : Nat

/-- The probability of rolling a green face on an octahedral die -/
def green_face_probability (die : OctahedralDie) : Rat :=
  die.green_faces / die.total_faces

/-- Theorem: The probability of rolling a green face on an octahedral die
    with 5 green faces and 3 yellow faces is 5/8 -/
theorem green_face_probability_octahedral_die :
  let die : OctahedralDie := {
    total_faces := 8,
    green_faces := 5,
    yellow_faces := 3
  }
  green_face_probability die = 5 / 8 := by
  sorry

end green_face_probability_octahedral_die_l967_96782


namespace x_intercept_of_line_l967_96725

/-- The x-intercept of the line 4x + 7y = 28 is the point (7,0) -/
theorem x_intercept_of_line (x y : ℚ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by sorry

end x_intercept_of_line_l967_96725


namespace right_triangle_hypotenuse_l967_96733

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
  c = 25 * Real.sqrt 2 := by
  sorry

end right_triangle_hypotenuse_l967_96733


namespace positive_numbers_l967_96723

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (sum_products_positive : a * b + b * c + c * a > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end positive_numbers_l967_96723


namespace fourteenSidedFigure_area_l967_96742

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon defined by a list of vertices -/
def Polygon := List Point

/-- The fourteen-sided figure described in the problem -/
def fourteenSidedFigure : Polygon := [
  ⟨1, 2⟩, ⟨2, 3⟩, ⟨4, 3⟩, ⟨5, 4⟩, ⟨5, 6⟩, ⟨6, 7⟩, ⟨7, 6⟩, ⟨7, 4⟩,
  ⟨6, 3⟩, ⟨4, 3⟩, ⟨3, 2⟩, ⟨3, 1⟩, ⟨2, 1⟩, ⟨1, 2⟩
]

/-- Calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℝ :=
  sorry -- Actual implementation would go here

/-- Theorem: The area of the fourteen-sided figure is 14 cm² -/
theorem fourteenSidedFigure_area :
  calculateArea fourteenSidedFigure = 14 := by
  sorry -- Proof would go here

end fourteenSidedFigure_area_l967_96742


namespace circle_intersection_equality_l967_96775

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary relations and functions
variable (on_circle : Point → Circle → Prop)
variable (center : Circle → Point)
variable (intersect : Circle → Circle → Point × Point)
variable (line_intersect : Point → Point → Circle → Point)
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem circle_intersection_equality 
  (circle1 circle2 : Circle) 
  (O P Q C A B : Point) :
  on_circle O circle1 ∧ 
  center circle2 = O ∧
  intersect circle1 circle2 = (P, Q) ∧
  on_circle C circle1 ∧
  line_intersect C P circle2 = A ∧
  line_intersect C Q circle2 = B →
  distance A B = distance P Q :=
by sorry

end circle_intersection_equality_l967_96775


namespace pure_imaginary_condition_l967_96740

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I * (a^2 - a - 2) = (a^2 - 2*a) + Complex.I * (a^2 - a - 2)) → a = 0 := by
  sorry

end pure_imaginary_condition_l967_96740


namespace greatest_possible_average_speed_l967_96751

/-- Represents a palindromic number --/
def IsPalindrome (n : ℕ) : Prop := sorry

/-- Calculates the next palindrome after a given number --/
def NextPalindrome (n : ℕ) : ℕ := sorry

/-- Represents the maximum speed limit in miles per hour --/
def MaxSpeedLimit : ℕ := 80

/-- Represents the trip duration in hours --/
def TripDuration : ℕ := 4

/-- Represents the initial odometer reading --/
def InitialReading : ℕ := 12321

theorem greatest_possible_average_speed :
  ∃ (finalReading : ℕ),
    IsPalindrome InitialReading ∧
    IsPalindrome finalReading ∧
    finalReading > InitialReading ∧
    finalReading ≤ InitialReading + MaxSpeedLimit * TripDuration ∧
    (∀ (n : ℕ),
      IsPalindrome n ∧
      n > InitialReading ∧
      n ≤ InitialReading + MaxSpeedLimit * TripDuration →
      n ≤ finalReading) ∧
    (finalReading - InitialReading) / TripDuration = 75 :=
by sorry

end greatest_possible_average_speed_l967_96751


namespace coin_sum_theorem_l967_96719

def coin_values : List Nat := [5, 10, 25, 50]

def is_valid_sum (n : Nat) : Prop :=
  ∃ (a b c d e : Nat), a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ d ∈ coin_values ∧ e ∈ coin_values ∧
    a + b + c + d + e = n

theorem coin_sum_theorem :
  ¬(is_valid_sum 40) ∧ 
  (is_valid_sum 65) ∧ 
  (is_valid_sum 85) ∧ 
  (is_valid_sum 105) ∧ 
  (is_valid_sum 130) := by
  sorry

end coin_sum_theorem_l967_96719


namespace root_product_property_l967_96710

theorem root_product_property (a b : ℂ) : 
  (a^4 + a^3 - 1 = 0) → (b^4 + b^3 - 1 = 0) → 
  ((a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0) := by
  sorry

end root_product_property_l967_96710


namespace triangle_area_and_square_coverage_l967_96716

/-- Given a triangle with side lengths 9, 40, and 41, prove its area and the fraction it covers of a square with side length 41. -/
theorem triangle_area_and_square_coverage :
  ∃ (triangle_area : ℝ) (square_area : ℝ) (coverage_fraction : ℚ),
    triangle_area = 180 ∧
    square_area = 41 ^ 2 ∧
    coverage_fraction = 180 / 1681 ∧
    (9 : ℝ) ^ 2 + 40 ^ 2 = 41 ^ 2 ∧
    triangle_area = (1 / 2 : ℝ) * 9 * 40 ∧
    coverage_fraction = triangle_area / square_area := by
  sorry

end triangle_area_and_square_coverage_l967_96716


namespace towel_shrinkage_l967_96736

theorem towel_shrinkage (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  let new_length := 0.8 * L
  let new_area := 0.64 * (L * B)
  ∃ new_breadth : ℝ, 
    new_breadth = 0.8 * B ∧ 
    new_length * new_breadth = new_area := by
  sorry

end towel_shrinkage_l967_96736


namespace m_mod_1000_l967_96786

/-- The set of integers from 1 to 12 -/
def T : Finset ℕ := Finset.range 12

/-- The number of sets of two non-empty disjoint subsets of T -/
def m : ℕ := (3^12 - 2 * 2^12 + 1) / 2

/-- Theorem stating that the remainder of m divided by 1000 is 625 -/
theorem m_mod_1000 : m % 1000 = 625 := by
  sorry

end m_mod_1000_l967_96786


namespace hyperbola_condition_l967_96783

theorem hyperbola_condition (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (m + 2) - y^2 / (m - 1) = 1) → (m < -2 ∨ m > 1) := by
  sorry

end hyperbola_condition_l967_96783


namespace y_derivative_l967_96748

noncomputable def y (x : ℝ) : ℝ := -2 * Real.exp x * Real.sin x

theorem y_derivative (x : ℝ) : 
  deriv y x = -2 * Real.exp x * (Real.cos x + Real.sin x) := by sorry

end y_derivative_l967_96748


namespace inverse_cube_squared_l967_96799

theorem inverse_cube_squared : (3⁻¹)^2 = (1 : ℚ) / 9 := by sorry

end inverse_cube_squared_l967_96799


namespace songs_downloaded_l967_96758

theorem songs_downloaded (internet_speed : ℕ) (song_size : ℕ) (download_time : ℕ) : 
  internet_speed = 20 → 
  song_size = 5 → 
  download_time = 1800 → 
  (internet_speed * download_time) / song_size = 7200 :=
by
  sorry

end songs_downloaded_l967_96758


namespace baking_contest_votes_l967_96774

theorem baking_contest_votes (witch_votes dragon_votes unicorn_votes : ℕ) : 
  witch_votes = 7 →
  unicorn_votes = 3 * witch_votes →
  dragon_votes > witch_votes →
  witch_votes + unicorn_votes + dragon_votes = 60 →
  dragon_votes - witch_votes = 25 := by
sorry

end baking_contest_votes_l967_96774


namespace task_completion_time_l967_96746

/-- Given two workers can complete a task in 35 days, and one worker can complete it in 60 days,
    prove that the other worker can complete the task in 84 days. -/
theorem task_completion_time (total_time : ℝ) (worker1_time : ℝ) (worker2_time : ℝ) : 
  (1 / total_time = 1 / worker1_time + 1 / worker2_time) →
  (total_time = 35) →
  (worker1_time = 60) →
  (worker2_time = 84) := by
sorry

end task_completion_time_l967_96746


namespace base_conversion_2345_to_base_7_l967_96755

theorem base_conversion_2345_to_base_7 :
  (2345 : ℕ) = 6 * (7 : ℕ)^3 + 5 * (7 : ℕ)^2 + 6 * (7 : ℕ)^1 + 0 * (7 : ℕ)^0 :=
by sorry

end base_conversion_2345_to_base_7_l967_96755


namespace quotient_base4_l967_96724

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a decimal number to its base 4 representation -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec helper (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else helper (m / 4) ((m % 4) :: acc)
  helper n []

/-- Theorem: The quotient of 1213₄ divided by 13₄ is equal to 32₄ -/
theorem quotient_base4 :
  let a := base4ToDecimal [3, 1, 2, 1]  -- 1213₄
  let b := base4ToDecimal [3, 1]        -- 13₄
  decimalToBase4 (a / b) = [2, 3]       -- 32₄
  := by sorry

end quotient_base4_l967_96724


namespace quadratic_equation_with_opposite_roots_l967_96711

theorem quadratic_equation_with_opposite_roots (x y : ℝ) :
  x^2 - 6*x + 9 = -|y - 1| →
  ∃ (a b c : ℝ), a ≠ 0 ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 ∧
  a = 1 ∧ b = -4 ∧ c = -3 :=
by sorry

end quadratic_equation_with_opposite_roots_l967_96711


namespace chocolate_box_theorem_l967_96718

/-- Represents a box of chocolates -/
structure ChocolateBox where
  initial_count : ℕ
  rows : ℕ
  columns : ℕ

/-- The state of the box after each rearrangement -/
inductive BoxState
  | Initial
  | AfterFirstRearrange
  | AfterSecondRearrange
  | Final

/-- Function to calculate the number of chocolates at each state -/
def chocolates_at_state (box : ChocolateBox) (state : BoxState) : ℕ :=
  match state with
  | BoxState.Initial => box.initial_count
  | BoxState.AfterFirstRearrange => 3 * box.columns - 1
  | BoxState.AfterSecondRearrange => 5 * box.rows - 1
  | BoxState.Final => box.initial_count / 3

theorem chocolate_box_theorem (box : ChocolateBox) :
  chocolates_at_state box BoxState.Initial = 60 ∧
  chocolates_at_state box BoxState.Initial - chocolates_at_state box BoxState.AfterFirstRearrange = 25 :=
by sorry


end chocolate_box_theorem_l967_96718


namespace average_sequence_l967_96798

theorem average_sequence (x : ℚ) : 
  (List.sum (List.range 149) + x) / 150 = 50 * x → x = 11175 / 7499 := by
  sorry

end average_sequence_l967_96798


namespace oil_division_l967_96762

/-- Proves that given 12.4 liters of oil divided into two bottles, where the large bottle can hold 2.6 liters more than the small bottle, the large bottle will hold 7.5 liters. -/
theorem oil_division (total_oil : ℝ) (difference : ℝ) (large_bottle : ℝ) : 
  total_oil = 12.4 →
  difference = 2.6 →
  large_bottle = (total_oil + difference) / 2 →
  large_bottle = 7.5 :=
by
  sorry

end oil_division_l967_96762


namespace ratio_of_numbers_l967_96763

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (h4 : a + b = 7 * (a - b) + 14) : a / b = 4 / 3 := by
  sorry

end ratio_of_numbers_l967_96763


namespace balls_without_holes_l967_96744

theorem balls_without_holes 
  (total_soccer : ℕ) 
  (total_basketball : ℕ) 
  (soccer_with_holes : ℕ) 
  (basketball_with_holes : ℕ) 
  (h1 : total_soccer = 40) 
  (h2 : total_basketball = 15) 
  (h3 : soccer_with_holes = 30) 
  (h4 : basketball_with_holes = 7) : 
  (total_soccer - soccer_with_holes) + (total_basketball - basketball_with_holes) = 18 :=
by
  sorry


end balls_without_holes_l967_96744


namespace stating_smallest_n_with_constant_term_l967_96721

/-- 
Given a positive integer n and the expression (4x^3 + 1/x^2)^n,
this function returns true if there exists a constant term in the expansion,
and false otherwise.
-/
def has_constant_term (n : ℕ+) : Prop :=
  ∃ r : ℕ, r ≤ n ∧ 3 * n = 5 * r

/-- 
Theorem stating that 5 is the smallest positive integer n 
for which there exists a constant term in the expansion of (4x^3 + 1/x^2)^n.
-/
theorem smallest_n_with_constant_term : 
  (∀ m : ℕ+, m < 5 → ¬has_constant_term m) ∧ has_constant_term 5 :=
sorry

end stating_smallest_n_with_constant_term_l967_96721


namespace quadratic_coefficient_l967_96772

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_coefficient (a b c : ℤ) :
  (∀ x, QuadraticFunction a b c x = a * (x - 2)^2 + 5) →
  QuadraticFunction a b c 1 = 2 →
  a = -3 := by sorry

end quadratic_coefficient_l967_96772


namespace part1_part2_l967_96741

-- Define the sequences
def a : ℕ → ℝ := λ n => 2^n
def b : ℕ → ℝ := λ n => 3^n
def c : ℕ → ℝ := λ n => a n + b n

-- Part 1
theorem part1 (p : ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, c (n + 2) - p * c (n + 1) = r * (c (n + 1) - p * c n)) →
  p = 2 ∨ p = 3 := by sorry

-- Part 2
theorem part2 {q1 q2 : ℝ} (hq : q1 ≠ q2) 
  (ha : ∀ n : ℕ, a (n + 1) = q1 * a n) 
  (hb : ∀ n : ℕ, b (n + 1) = q2 * b n) :
  ¬ (∃ r : ℝ, ∀ n : ℕ, c (n + 1) = r * c n) := by sorry

end part1_part2_l967_96741


namespace quadratic_inequality_solution_l967_96750

theorem quadratic_inequality_solution (n : ℕ) (x : ℝ) :
  (∀ n : ℕ, n^2 * x^2 - (2*n^2 + n) * x + n^2 + n - 6 ≤ 0) ↔ x = 1 := by
  sorry

end quadratic_inequality_solution_l967_96750


namespace tangent_line_a_value_l967_96734

/-- A line in polar coordinates tangent to a circle. -/
structure PolarLineTangentToCircle where
  a : ℝ
  tangent_line : ℝ → ℝ → Prop
  circle : ℝ → ℝ → Prop
  a_positive : a > 0
  is_tangent : ∀ θ ρ, tangent_line ρ θ ↔ ρ * (Real.cos θ + Real.sin θ) = a
  circle_eq : ∀ θ ρ, circle ρ θ ↔ ρ = 2 * Real.cos θ

/-- The value of 'a' for a line tangent to the given circle is 1 + √2. -/
theorem tangent_line_a_value (h : PolarLineTangentToCircle) : h.a = 1 + Real.sqrt 2 := by
  sorry

end tangent_line_a_value_l967_96734


namespace salary_problem_l967_96767

theorem salary_problem (A_salary B_salary : ℝ) 
  (h1 : A_salary = 4500)
  (h2 : A_salary * 0.05 = B_salary * 0.15)
  : A_salary + B_salary = 6000 := by
  sorry

end salary_problem_l967_96767


namespace grapes_distribution_l967_96787

theorem grapes_distribution (total_grapes : ℕ) (num_kids : ℕ) 
  (h1 : total_grapes = 50)
  (h2 : num_kids = 7) :
  total_grapes % num_kids = 1 := by
  sorry

end grapes_distribution_l967_96787


namespace conditional_without_else_l967_96727

-- Define the structure of conditional statements
inductive ConditionalStatement
  | ifThen : ConditionalStatement
  | ifThenElse : ConditionalStatement

-- Define a property for conditional statements with one branch
def hasOneBranch (stmt : ConditionalStatement) : Prop :=
  match stmt with
  | ConditionalStatement.ifThen => true
  | ConditionalStatement.ifThenElse => false

-- Theorem: A conditional statement can be without the statement after ELSE
theorem conditional_without_else :
  ∃ (stmt : ConditionalStatement), hasOneBranch stmt ∧ stmt = ConditionalStatement.ifThen :=
sorry

end conditional_without_else_l967_96727


namespace train_length_is_60_l967_96766

/-- Two trains with equal length on parallel tracks -/
structure TrainSystem where
  train_length : ℝ
  fast_speed : ℝ
  slow_speed : ℝ
  passing_time : ℝ

/-- The train system satisfies the given conditions -/
def valid_train_system (ts : TrainSystem) : Prop :=
  ts.fast_speed = 72 * (5/18) ∧  -- 72 km/h in m/s
  ts.slow_speed = 54 * (5/18) ∧  -- 54 km/h in m/s
  ts.passing_time = 24

/-- Theorem stating that the length of each train is 60 meters -/
theorem train_length_is_60 (ts : TrainSystem) 
  (h : valid_train_system ts) : ts.train_length = 60 := by
  sorry

#check train_length_is_60

end train_length_is_60_l967_96766


namespace triangle_to_hexagon_area_ratio_l967_96717

/-- A regular hexagon with an inscribed equilateral triangle -/
structure RegularHexagonWithTriangle where
  -- The area of the regular hexagon
  hexagon_area : ℝ
  -- The area of the inscribed equilateral triangle
  triangle_area : ℝ

/-- The ratio of the inscribed triangle's area to the hexagon's area is 1/6 -/
theorem triangle_to_hexagon_area_ratio 
  (hex : RegularHexagonWithTriangle) : 
  hex.triangle_area / hex.hexagon_area = 1 / 6 := by
  sorry

end triangle_to_hexagon_area_ratio_l967_96717
