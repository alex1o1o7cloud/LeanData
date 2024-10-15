import Mathlib

namespace NUMINAMATH_CALUDE_trigonometric_identities_l2116_211642

theorem trigonometric_identities (θ : Real) 
  (h : (4 * Real.sin θ - 2 * Real.cos θ) / (3 * Real.sin θ + 5 * Real.cos θ) = 6 / 11) : 
  (Real.tan θ = 2) ∧ 
  ((5 * (Real.cos θ)^2) / (Real.sin (2*θ) + 2 * Real.sin θ * Real.cos θ - 3 * (Real.cos θ)^2) = 1) ∧ 
  (1 - 4 * Real.sin θ * Real.cos θ + 2 * (Real.cos θ)^2 = -1/5) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2116_211642


namespace NUMINAMATH_CALUDE_replaced_person_weight_l2116_211639

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  new_person_weight - initial_count * average_increase

/-- Theorem stating the weight of the replaced person under the given conditions -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 6 93 = 45 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l2116_211639


namespace NUMINAMATH_CALUDE_karen_start_time_l2116_211682

/-- Proves that Karen starts the race 4 minutes late given the specified conditions. -/
theorem karen_start_time (karen_speed tom_speed : ℝ) (tom_distance : ℝ) (karen_lead : ℝ) : 
  karen_speed = 60 →
  tom_speed = 45 →
  tom_distance = 24 →
  karen_lead = 4 →
  (tom_distance / tom_speed - (tom_distance + karen_lead) / karen_speed) * 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_karen_start_time_l2116_211682


namespace NUMINAMATH_CALUDE_product_unit_digit_l2116_211686

-- Define a function to get the unit digit of a number
def unitDigit (n : ℕ) : ℕ := n % 10

-- Define the numbers given in the problem
def a : ℕ := 7858
def b : ℕ := 1086
def c : ℕ := 4582
def d : ℕ := 9783

-- State the theorem
theorem product_unit_digit :
  unitDigit (a * b * c * d) = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_unit_digit_l2116_211686


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l2116_211645

theorem positive_numbers_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hnot_all_equal : ¬(a = b ∧ b = c)) : 
  ((a - b)^2 + (b - c)^2 + (c - a)^2 ≠ 0) ∧ 
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l2116_211645


namespace NUMINAMATH_CALUDE_sphere_radius_from_intersection_l2116_211685

theorem sphere_radius_from_intersection (width depth : ℝ) (h_width : width = 30) (h_depth : depth = 10) :
  let r := Real.sqrt ((width / 2) ^ 2 + (width / 4 + depth) ^ 2)
  ∃ ε > 0, abs (r - 22.1129) < ε :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_from_intersection_l2116_211685


namespace NUMINAMATH_CALUDE_orange_packing_problem_l2116_211676

/-- Given a total number of oranges and the capacity of each box, 
    calculate the number of boxes needed to pack all oranges. -/
def boxes_needed (total_oranges : ℕ) (oranges_per_box : ℕ) : ℕ :=
  total_oranges / oranges_per_box

/-- Theorem stating that 265 boxes are needed to pack 2650 oranges
    when each box holds 10 oranges. -/
theorem orange_packing_problem :
  boxes_needed 2650 10 = 265 := by
  sorry

end NUMINAMATH_CALUDE_orange_packing_problem_l2116_211676


namespace NUMINAMATH_CALUDE_prime_power_divisibility_l2116_211695

theorem prime_power_divisibility (p n : ℕ) (hp : Prime p) (h : p ∣ n^2020) : p^2020 ∣ n^2020 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_divisibility_l2116_211695


namespace NUMINAMATH_CALUDE_cone_volume_ratio_l2116_211647

/-- Given two sectors of a circle with central angles in the ratio 3:4, 
    the ratio of the volumes of the cones formed by rolling these sectors is 27:64 -/
theorem cone_volume_ratio (r : ℝ) (θ : ℝ) (h₁ h₂ : ℝ) :
  r > 0 → θ > 0 →
  3 * θ + 4 * θ = 2 * π →
  h₁ = Real.sqrt (r^2 - (3 * θ * r / (2 * π))^2) →
  h₂ = Real.sqrt (r^2 - (4 * θ * r / (2 * π))^2) →
  (1/3 * π * (3 * θ * r / (2 * π))^2 * h₁) / (1/3 * π * (4 * θ * r / (2 * π))^2 * h₂) = 27/64 :=
by sorry

#check cone_volume_ratio

end NUMINAMATH_CALUDE_cone_volume_ratio_l2116_211647


namespace NUMINAMATH_CALUDE_expand_expression_l2116_211688

theorem expand_expression (x y : ℝ) : (10 * x - 6 * y + 9) * (3 * y) = 30 * x * y - 18 * y^2 + 27 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2116_211688


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l2116_211697

theorem shaded_fraction_of_rectangle (length width : ℕ) (shaded_area : ℚ) :
  length = 15 →
  width = 20 →
  shaded_area = (1 / 2 : ℚ) * (1 / 4 : ℚ) * (length * width : ℚ) →
  shaded_area / (length * width : ℚ) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l2116_211697


namespace NUMINAMATH_CALUDE_max_ratio_square_extension_l2116_211657

/-- Given a square ABCD with side length a, the ratio MA:MB is maximized 
    when M is positioned on the extension of CD such that MC = 2a / (1 + √5) -/
theorem max_ratio_square_extension (a : ℝ) (h : a > 0) :
  let square := {A : ℝ × ℝ | A.1 ∈ [0, a] ∧ A.2 ∈ [0, a]}
  let C := (a, 0)
  let D := (a, a)
  let M (x : ℝ) := (a + x, 0)
  let ratio (x : ℝ) := ‖M x - (0, a)‖ / ‖M x - (a, a)‖
  ∃ (x_max : ℝ), x_max = 2 * a / (1 + Real.sqrt 5) ∧
    ∀ (x : ℝ), x > 0 → ratio x ≤ ratio x_max :=
by
  sorry


end NUMINAMATH_CALUDE_max_ratio_square_extension_l2116_211657


namespace NUMINAMATH_CALUDE_M_eq_roster_l2116_211674

def M : Set ℚ := {x | ∃ (m : ℤ) (n : ℕ), n > 0 ∧ n ≤ 3 ∧ abs m < 2 ∧ x = m / n}

theorem M_eq_roster : M = {-1, -1/2, -1/3, 0, 1/3, 1/2, 1} := by sorry

end NUMINAMATH_CALUDE_M_eq_roster_l2116_211674


namespace NUMINAMATH_CALUDE_harmonic_division_in_circumscribed_square_l2116_211606

-- Define the square
structure Square where
  side : ℝ
  center : ℝ × ℝ

-- Define the circle
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the configuration
structure Configuration where
  square : Square
  circle : Circle
  tangent : Line
  P : Point
  Q : Point
  R : Point
  S : Point

-- Define the property of being circumscribed
def is_circumscribed (s : Square) (c : Circle) : Prop :=
  s.side = 2 * c.radius ∧ s.center = c.center

-- Define the property of being a tangent
def is_tangent (l : Line) (c : Circle) : Prop :=
  ∃ (p : Point), p.x^2 + p.y^2 = c.radius^2 ∧ l.a * p.x + l.b * p.y + l.c = 0

-- Define the property of points being on the square or its extensions
def on_square_or_extension (p : Point) (s : Square) : Prop :=
  (p.x = s.center.1 - s.side/2 ∨ p.x = s.center.1 + s.side/2) ∨
  (p.y = s.center.2 - s.side/2 ∨ p.y = s.center.2 + s.side/2)

-- Define the harmonic division property
def harmonic_division (a : Point) (b : Point) (c : Point) (d : Point) : Prop :=
  (a.x - c.x) / (b.x - c.x) = (a.x - d.x) / (b.x - d.x)

-- Main theorem
theorem harmonic_division_in_circumscribed_square (cfg : Configuration) 
  (h1 : is_circumscribed cfg.square cfg.circle)
  (h2 : is_tangent cfg.tangent cfg.circle)
  (h3 : on_square_or_extension cfg.P cfg.square)
  (h4 : on_square_or_extension cfg.Q cfg.square)
  (h5 : on_square_or_extension cfg.R cfg.square)
  (h6 : on_square_or_extension cfg.S cfg.square) :
  harmonic_division cfg.P cfg.R cfg.Q cfg.S ∧ harmonic_division cfg.Q cfg.S cfg.P cfg.R :=
sorry

end NUMINAMATH_CALUDE_harmonic_division_in_circumscribed_square_l2116_211606


namespace NUMINAMATH_CALUDE_harrys_annual_pet_feeding_cost_l2116_211627

/-- Calculates the annual cost of feeding pets given the number of each type and their monthly feeding costs. -/
def annual_pet_feeding_cost (num_geckos num_iguanas num_snakes : ℕ) 
                            (gecko_cost iguana_cost snake_cost : ℕ) : ℕ :=
  12 * (num_geckos * gecko_cost + num_iguanas * iguana_cost + num_snakes * snake_cost)

/-- Theorem stating that Harry's annual pet feeding cost is $1140. -/
theorem harrys_annual_pet_feeding_cost : 
  annual_pet_feeding_cost 3 2 4 15 5 10 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_harrys_annual_pet_feeding_cost_l2116_211627


namespace NUMINAMATH_CALUDE_cycle_original_price_l2116_211622

/-- Proves that given a cycle sold for 1260 with a 40% gain, the original price was 900 --/
theorem cycle_original_price (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 1260)
  (h2 : gain_percentage = 40) : 
  (selling_price / (1 + gain_percentage / 100)) = 900 := by
  sorry

end NUMINAMATH_CALUDE_cycle_original_price_l2116_211622


namespace NUMINAMATH_CALUDE_tim_and_linda_mowing_time_l2116_211614

/-- The time it takes for two people to complete a task together, given their individual rates -/
def combined_time (rate1 rate2 : ℚ) : ℚ :=
  1 / (rate1 + rate2)

/-- Proof that Tim and Linda can mow the lawn together in 6/7 hours -/
theorem tim_and_linda_mowing_time :
  let tim_rate : ℚ := 1 / (3/2)  -- Tim's rate: 1 lawn per 1.5 hours
  let linda_rate : ℚ := 1 / 2    -- Linda's rate: 1 lawn per 2 hours
  combined_time tim_rate linda_rate = 6/7 := by
  sorry

#eval (combined_time (1 / (3/2)) (1 / 2))

end NUMINAMATH_CALUDE_tim_and_linda_mowing_time_l2116_211614


namespace NUMINAMATH_CALUDE_remainder_a_squared_minus_3b_l2116_211638

theorem remainder_a_squared_minus_3b (a b : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 5) 
  (h_ineq : a^2 > 3*b) : 
  (a^2 - 3*b) % 7 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_a_squared_minus_3b_l2116_211638


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l2116_211626

theorem solution_satisfies_equations :
  ∃ (j k : ℚ), 5 * j - 42 * k = 1 ∧ 2 * k - j = 3 ∧ j = -4 ∧ k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l2116_211626


namespace NUMINAMATH_CALUDE_color_copies_proof_l2116_211672

/-- The price per color copy at print shop X -/
def price_x : ℚ := 1.25

/-- The price per color copy at print shop Y -/
def price_y : ℚ := 2.75

/-- The difference in total charge between print shop Y and X -/
def charge_difference : ℚ := 120

/-- The number of color copies -/
def num_copies : ℚ := 80

theorem color_copies_proof :
  price_y * num_copies = price_x * num_copies + charge_difference :=
by sorry

end NUMINAMATH_CALUDE_color_copies_proof_l2116_211672


namespace NUMINAMATH_CALUDE_hyperbola_intersection_ratio_difference_l2116_211649

/-- The hyperbola with equation x²/2 - y²/2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2/2 - y^2/2 = 1

/-- The right branch of the hyperbola -/
def right_branch (x y : ℝ) : Prop := hyperbola x y ∧ x > 0

/-- Point P lies on the right branch of the hyperbola -/
def P_on_right_branch (P : ℝ × ℝ) : Prop := right_branch P.1 P.2

/-- Point A is the intersection of PF₁ and the hyperbola -/
def A_is_intersection (P A : ℝ × ℝ) : Prop :=
  hyperbola A.1 A.2 ∧ ∃ t : ℝ, A = (t * (P.1 + 2) - 2, t * P.2)

/-- Point B is the intersection of PF₂ and the hyperbola -/
def B_is_intersection (P B : ℝ × ℝ) : Prop :=
  hyperbola B.1 B.2 ∧ ∃ t : ℝ, B = (t * (P.1 - 2) + 2, t * P.2)

/-- The main theorem -/
theorem hyperbola_intersection_ratio_difference (P A B : ℝ × ℝ) :
  P_on_right_branch P →
  A_is_intersection P A →
  B_is_intersection P B →
  ∃ (PF₁ AF₁ PF₂ BF₂ : ℝ),
    PF₁ / AF₁ - PF₂ / BF₂ = 6 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_ratio_difference_l2116_211649


namespace NUMINAMATH_CALUDE_messenger_speed_l2116_211648

/-- Messenger speed problem -/
theorem messenger_speed (team_length : ℝ) (team_speed : ℝ) (total_time : ℝ) :
  team_length = 6 →
  team_speed = 5 →
  total_time = 0.5 →
  ∃ messenger_speed : ℝ,
    messenger_speed > 0 ∧
    (team_length / (messenger_speed + team_speed) + team_length / (messenger_speed - team_speed) = total_time) ∧
    messenger_speed = 25 := by
  sorry

end NUMINAMATH_CALUDE_messenger_speed_l2116_211648


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_l2116_211613

/-- Given an isosceles triangle ABC with AB = BC = a and AC = b, 
    if ax² - √2·bx + a = 0 has two real roots with absolute difference √2,
    then ∠ABC = 120° -/
theorem isosceles_triangle_angle (a b : ℝ) (u : ℝ) : 
  a > 0 → b > 0 →
  (∃ x y : ℝ, x ≠ y ∧ 
    a * x^2 - Real.sqrt 2 * b * x + a = 0 ∧ 
    a * y^2 - Real.sqrt 2 * b * y + a = 0 ∧
    |x - y| = Real.sqrt 2) →
  b^2 = 2 * a^2 * (1 - Real.cos u) →
  u = 120 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_l2116_211613


namespace NUMINAMATH_CALUDE_equal_x_y_l2116_211679

theorem equal_x_y (x y z : ℝ) (h1 : x = 6 - y) (h2 : z^2 = x*y - 9) : x = y := by
  sorry

end NUMINAMATH_CALUDE_equal_x_y_l2116_211679


namespace NUMINAMATH_CALUDE_parabola_decreasing_right_of_axis_l2116_211631

-- Define the parabola function
def f (b c x : ℝ) : ℝ := -x^2 + b*x + c

-- State the theorem
theorem parabola_decreasing_right_of_axis (b c : ℝ) :
  (∀ x, f b c x = f b c (6 - x)) →  -- Axis of symmetry at x = 3
  ∀ x > 3, ∀ y > x, f b c y < f b c x :=
sorry

end NUMINAMATH_CALUDE_parabola_decreasing_right_of_axis_l2116_211631


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_minus_one_l2116_211662

theorem imaginary_part_of_i_minus_one (i : ℂ) (h : i * i = -1) :
  Complex.im (i - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_minus_one_l2116_211662


namespace NUMINAMATH_CALUDE_chime_2003_date_l2116_211619

/-- Represents a date with year, month, and day -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Represents a time with hour and minute -/
structure Time where
  hour : Nat
  minute : Nat

/-- Calculates the number of chimes for a given hour -/
def hourChimes (hour : Nat) : Nat :=
  hour % 12

/-- Calculates the total number of chimes from a start date and time to an end date -/
def totalChimes (startDate : Date) (startTime : Time) (endDate : Date) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem chime_2003_date :
  let startDate := Date.mk 2003 2 28
  let startTime := Time.mk 15 15
  let endDate := Date.mk 2003 3 22
  totalChimes startDate startTime endDate = 2003 :=
sorry

end NUMINAMATH_CALUDE_chime_2003_date_l2116_211619


namespace NUMINAMATH_CALUDE_angle_bisector_ratio_l2116_211607

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define points D and E on AB and AC respectively
def D (triangle : Triangle) : ℝ × ℝ := sorry
def E (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the angle bisector AT
def T (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the intersection point F of AT and DE
def F (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the lengths
def AD (triangle : Triangle) : ℝ := 2
def DB (triangle : Triangle) : ℝ := 6
def AE (triangle : Triangle) : ℝ := 5
def EC (triangle : Triangle) : ℝ := 3

-- Define the ratio AF/AT
def AF_AT_ratio (triangle : Triangle) : ℝ := sorry

-- Theorem statement
theorem angle_bisector_ratio (triangle : Triangle) :
  AF_AT_ratio triangle = 2/5 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_ratio_l2116_211607


namespace NUMINAMATH_CALUDE_password_probability_l2116_211610

-- Define the set of possible last digits
def LastDigits : Finset Char := {'A', 'a', 'B', 'b'}

-- Define the set of possible second-to-last digits
def SecondLastDigits : Finset Nat := {4, 5, 6}

-- Define the type for a password
def Password := Nat × Char

-- Define the set of all possible passwords
def AllPasswords : Finset Password :=
  SecondLastDigits.product LastDigits

-- Theorem statement
theorem password_probability :
  (Finset.card AllPasswords : ℚ) = 12 ∧
  (1 : ℚ) / (Finset.card AllPasswords : ℚ) = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_password_probability_l2116_211610


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_zero_l2116_211640

theorem ceiling_floor_sum_zero : 
  Int.ceil (7 / 3 : ℚ) + Int.floor (-(7 / 3) : ℚ) + 
  Int.ceil (4 / 5 : ℚ) + Int.floor (-(4 / 5) : ℚ) = 0 := by
sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_zero_l2116_211640


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2116_211655

-- Define the equations
def equation1 (x : ℝ) : Prop := 4 - x = 3 * (2 - x)
def equation2 (x : ℝ) : Prop := (2 * x - 1) / 2 - (2 * x + 5) / 3 = (6 * x - 1) / 6 - 1

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1 := by sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -1.5 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2116_211655


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2116_211625

theorem complex_equation_solution (a b : ℝ) 
  (h : (a - 1 : ℂ) + a * I = 3 + 2 * b * I) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2116_211625


namespace NUMINAMATH_CALUDE_nested_bracket_evaluation_l2116_211635

def bracket (a b c : ℚ) : ℚ := (a + b) / c

theorem nested_bracket_evaluation :
  let outer_bracket := bracket (bracket 72 36 108) (bracket 4 2 6) (bracket 12 6 18)
  outer_bracket = 2 := by sorry

end NUMINAMATH_CALUDE_nested_bracket_evaluation_l2116_211635


namespace NUMINAMATH_CALUDE_perpendicular_distance_is_four_l2116_211651

/-- A rectangular parallelepiped with vertices H, E, F, and G -/
structure Parallelepiped where
  H : ℝ × ℝ × ℝ
  E : ℝ × ℝ × ℝ
  F : ℝ × ℝ × ℝ
  G : ℝ × ℝ × ℝ

/-- The specific parallelepiped described in the problem -/
def specificParallelepiped : Parallelepiped :=
  { H := (0, 0, 0)
    E := (5, 0, 0)
    F := (0, 6, 0)
    G := (0, 0, 4) }

/-- The perpendicular distance from a point to a plane -/
noncomputable def perpendicularDistance (p : ℝ × ℝ × ℝ) (plane : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- The plane containing points E, F, and G -/
def planEFG (p : Parallelepiped) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The theorem to be proved -/
theorem perpendicular_distance_is_four :
  perpendicularDistance specificParallelepiped.H (planEFG specificParallelepiped) = 4 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_distance_is_four_l2116_211651


namespace NUMINAMATH_CALUDE_solution_product_theorem_l2116_211611

theorem solution_product_theorem (a b : ℝ) : 
  a ≠ b → 
  (a^4 + a^3 - 1 = 0) → 
  (b^4 + b^3 - 1 = 0) → 
  ((a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_solution_product_theorem_l2116_211611


namespace NUMINAMATH_CALUDE_fish_crate_weight_l2116_211687

theorem fish_crate_weight (total_weight : ℝ) (cost_per_crate : ℝ) (total_cost : ℝ)
  (h1 : total_weight = 540)
  (h2 : cost_per_crate = 1.5)
  (h3 : total_cost = 27) :
  total_weight / (total_cost / cost_per_crate) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_crate_weight_l2116_211687


namespace NUMINAMATH_CALUDE_little_red_riding_hood_waffles_l2116_211677

theorem little_red_riding_hood_waffles (initial_waffles : ℕ) : 
  (∃ (x : ℕ), 
    initial_waffles = 14 * x ∧ 
    (initial_waffles / 2 - x) / 2 - x = x ∧
    x > 0) →
  initial_waffles % 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_little_red_riding_hood_waffles_l2116_211677


namespace NUMINAMATH_CALUDE_op_properties_l2116_211629

-- Define the @ operation
def op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Theorem statement
theorem op_properties :
  (op 1 (-2) = -8) ∧ 
  (∀ a b : ℝ, op a b = op b a) ∧
  (∀ a b : ℝ, a + b = 0 → op a a + op b b = 8 * a^2) := by
sorry

end NUMINAMATH_CALUDE_op_properties_l2116_211629


namespace NUMINAMATH_CALUDE_digit_equation_proof_l2116_211692

theorem digit_equation_proof :
  ∀ (A B D : ℕ),
    A ≤ 9 → B ≤ 9 → D ≤ 9 →
    A ≥ B →
    (100 * A + 10 * B + D) * (A + B + D) = 1323 →
    D = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_proof_l2116_211692


namespace NUMINAMATH_CALUDE_triangle_problem_l2116_211652

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a > c →
  b = 3 →
  (a * c : ℝ) * (1 / 3 : ℝ) = 2 →  -- This represents BA · BC = 2 and cos B = 1/3
  a^2 + c^2 = b^2 + 2 * (a * c : ℝ) * (1 / 3 : ℝ) →  -- Law of cosines
  (a = 3 ∧ c = 2) ∧ 
  Real.cos (B - C) = 23 / 27 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2116_211652


namespace NUMINAMATH_CALUDE_negation_of_all_rectangles_equal_diagonals_l2116_211669

-- Define a type for rectangles
variable (Rectangle : Type)

-- Define a predicate for equal diagonals
variable (has_equal_diagonals : Rectangle → Prop)

-- Statement to prove
theorem negation_of_all_rectangles_equal_diagonals :
  (¬ ∀ r : Rectangle, has_equal_diagonals r) ↔ (∃ r : Rectangle, ¬ has_equal_diagonals r) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_all_rectangles_equal_diagonals_l2116_211669


namespace NUMINAMATH_CALUDE_jumping_competition_result_l2116_211612

/-- The difference in average jumps per minute between two competitors -/
def jump_difference (total_time : ℕ) (jumps_a : ℕ) (jumps_b : ℕ) : ℚ :=
  (jumps_a - jumps_b : ℚ) / total_time

theorem jumping_competition_result :
  jump_difference 5 480 420 = 12 := by
  sorry

end NUMINAMATH_CALUDE_jumping_competition_result_l2116_211612


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l2116_211675

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to get the last two nonzero digits
def lastTwoNonzeroDigits (n : ℕ) : ℕ :=
  n % 100

-- Theorem statement
theorem last_two_nonzero_digits_80_factorial :
  lastTwoNonzeroDigits (factorial 80) = 76 := by
  sorry


end NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l2116_211675


namespace NUMINAMATH_CALUDE_mistaken_calculation_l2116_211634

theorem mistaken_calculation (x : ℕ) : 
  423 - x = 421 → (423 * x) + 421 - 500 = 767 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_l2116_211634


namespace NUMINAMATH_CALUDE_magnitude_of_2a_plus_b_l2116_211608

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (0, -1, 1)
def b : ℝ × ℝ × ℝ := (1, 0, 1)

-- Define the operation 2a + b
def result : ℝ × ℝ × ℝ := (2 * a.1 + b.1, 2 * a.2.1 + b.2.1, 2 * a.2.2 + b.2.2)

-- Theorem statement
theorem magnitude_of_2a_plus_b : 
  Real.sqrt ((result.1)^2 + (result.2.1)^2 + (result.2.2)^2) = Real.sqrt 14 :=
by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_2a_plus_b_l2116_211608


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l2116_211605

def james_age_3_years_ago : ℕ := 27
def matt_current_age : ℕ := 65
def years_since_james_27 : ℕ := 3
def years_to_future : ℕ := 5

def james_future_age : ℕ := james_age_3_years_ago + years_since_james_27 + years_to_future
def matt_future_age : ℕ := matt_current_age + years_to_future

theorem age_ratio_is_two_to_one :
  (matt_future_age : ℚ) / james_future_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l2116_211605


namespace NUMINAMATH_CALUDE_term_properties_l2116_211699

-- Define a structure for a monomial term
structure Monomial where
  coefficient : ℚ
  x_power : ℕ
  y_power : ℕ

-- Define the monomial -1/3 * x * y^2
def term : Monomial := {
  coefficient := -1/3,
  x_power := 1,
  y_power := 2
}

-- Define the coefficient of a monomial
def coefficient (m : Monomial) : ℚ := m.coefficient

-- Define the degree of a monomial
def degree (m : Monomial) : ℕ := m.x_power + m.y_power

-- Theorem stating the coefficient and degree of the term
theorem term_properties :
  coefficient term = -1/3 ∧ degree term = 3 := by
  sorry


end NUMINAMATH_CALUDE_term_properties_l2116_211699


namespace NUMINAMATH_CALUDE_pie_chart_proportions_l2116_211609

theorem pie_chart_proportions :
  ∀ (white black gray blue : ℚ),
    white = 3 * black →
    black = 2 * gray →
    blue = gray →
    white + black + gray + blue = 1 →
    white = 3/5 ∧ black = 1/5 ∧ gray = 1/10 ∧ blue = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_proportions_l2116_211609


namespace NUMINAMATH_CALUDE_r_power_four_plus_inverse_l2116_211621

theorem r_power_four_plus_inverse (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_power_four_plus_inverse_l2116_211621


namespace NUMINAMATH_CALUDE_circle_equation_l2116_211600

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 2)

-- Define the radius of the circle
def radius : ℝ := 4

-- State the theorem
theorem circle_equation :
  ∀ (x y : ℝ), ((x + 1)^2 + (y - 2)^2 = 16) ↔ 
  ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2116_211600


namespace NUMINAMATH_CALUDE_bus_speed_l2116_211630

/-- The initial average speed of a bus given specific journey conditions -/
theorem bus_speed (D : ℝ) (h : D > 0) : ∃ v : ℝ,
  v > 0 ∧ 
  D = v * (65 / 60) ∧ 
  D = (v + 5) * 1 ∧
  v = 60 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_l2116_211630


namespace NUMINAMATH_CALUDE_binomial_18_10_l2116_211664

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 43758 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l2116_211664


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2116_211603

def A : Set Int := {1, 2}
def B : Set Int := {-1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2116_211603


namespace NUMINAMATH_CALUDE_parabola_equation_l2116_211646

/-- A parabola with vertex at the origin and axis of symmetry x = -4 has the standard equation y^2 = 16x -/
theorem parabola_equation (y x : ℝ) : 
  (∀ p : ℝ, p > 0 → y^2 = 2*p*x) → -- Standard form of parabola equation
  (∀ p : ℝ, -p/2 = -4) →           -- Axis of symmetry condition
  y^2 = 16*x :=                    -- Conclusion: standard equation
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2116_211646


namespace NUMINAMATH_CALUDE_problem_statement_l2116_211696

theorem problem_statement : 
  Real.sqrt 12 + |1 - Real.sqrt 3| + (π - 2023)^0 = 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2116_211696


namespace NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l2116_211698

/-- Given a cone whose lateral surface area is equal to the area of a semicircle with area 2π,
    prove that the volume of the cone is (√3/3)π. -/
theorem cone_volume_from_lateral_surface (cone : Real → Real → Real) 
  (lateral_surface_area : Real) (semicircle_area : Real) :
  lateral_surface_area = semicircle_area →
  semicircle_area = 2 * Real.pi →
  (∃ (r h : Real), cone r h = (1/3) * Real.pi * r^2 * h ∧ 
                   cone r h = (Real.sqrt 3 / 3) * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l2116_211698


namespace NUMINAMATH_CALUDE_at_least_one_angle_leq_60_l2116_211667

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : a + b + c = 180

-- Theorem statement
theorem at_least_one_angle_leq_60 (t : Triangle) : 
  t.a ≤ 60 ∨ t.b ≤ 60 ∨ t.c ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_angle_leq_60_l2116_211667


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l2116_211616

/-- Given a 6-8-10 right triangle with vertices as centers of three mutually externally tangent circles,
    the sum of the areas of these circles is 56π. -/
theorem sum_of_circle_areas (a b c : ℝ) (r s t : ℝ) : 
  a = 6 ∧ b = 8 ∧ c = 10 ∧  -- Triangle side lengths
  a^2 + b^2 = c^2 ∧         -- Right triangle condition
  r + s = a ∧              -- Circles are externally tangent
  r + t = b ∧
  s + t = c ∧
  r > 0 ∧ s > 0 ∧ t > 0 →   -- Radii are positive
  π * (r^2 + s^2 + t^2) = 56 * π :=
by sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l2116_211616


namespace NUMINAMATH_CALUDE_total_pages_read_l2116_211637

-- Define the book's properties
def total_pages : ℕ := 95
def total_chapters : ℕ := 8

-- Define Jake's reading
def initial_pages_read : ℕ := 37
def additional_pages_read : ℕ := 25

-- Theorem to prove
theorem total_pages_read :
  initial_pages_read + additional_pages_read = 62 :=
by sorry

end NUMINAMATH_CALUDE_total_pages_read_l2116_211637


namespace NUMINAMATH_CALUDE_problem_solution_l2116_211623

theorem problem_solution (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + e^2 + 1 = d + Real.sqrt (a + b + c + e - 2*d)) : 
  d = -23/8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2116_211623


namespace NUMINAMATH_CALUDE_fire_truck_ladder_height_l2116_211681

theorem fire_truck_ladder_height (distance_to_building : ℝ) (ladder_length : ℝ) :
  distance_to_building = 5 →
  ladder_length = 13 →
  ∃ (height : ℝ), height^2 + distance_to_building^2 = ladder_length^2 ∧ height = 12 :=
by sorry

end NUMINAMATH_CALUDE_fire_truck_ladder_height_l2116_211681


namespace NUMINAMATH_CALUDE_shortest_path_in_sqrt2_octahedron_l2116_211632

/-- A regular octahedron -/
structure RegularOctahedron where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

/-- The shortest path between midpoints of non-adjacent edges -/
def shortestPathBetweenMidpoints (o : RegularOctahedron) : ℝ :=
  sorry

/-- Theorem: In a regular octahedron with edge length √2, the shortest path
    between midpoints of non-adjacent edges is √2 -/
theorem shortest_path_in_sqrt2_octahedron :
  let o : RegularOctahedron := ⟨ Real.sqrt 2, sorry ⟩
  shortestPathBetweenMidpoints o = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_in_sqrt2_octahedron_l2116_211632


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fourth_power_l2116_211636

theorem magnitude_of_complex_fourth_power : 
  Complex.abs ((5 - 2 * Complex.I * Real.sqrt 3) ^ 4) = 1369 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fourth_power_l2116_211636


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l2116_211683

def total_balls : ℕ := 15
def red_balls : ℕ := 7
def blue_balls : ℕ := 8

theorem probability_two_red_balls :
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l2116_211683


namespace NUMINAMATH_CALUDE_target_miss_probability_l2116_211601

theorem target_miss_probability 
  (p_I p_II p_III : ℝ) 
  (h_I : p_I = 0.35) 
  (h_II : p_II = 0.30) 
  (h_III : p_III = 0.25) : 
  1 - (p_I + p_II + p_III) = 0.1 := by
sorry

end NUMINAMATH_CALUDE_target_miss_probability_l2116_211601


namespace NUMINAMATH_CALUDE_correct_outfit_count_l2116_211665

/-- The number of shirts -/
def num_shirts : ℕ := 5

/-- The number of pants -/
def num_pants : ℕ := 6

/-- The number of formal pants -/
def num_formal_pants : ℕ := 3

/-- The number of casual pants -/
def num_casual_pants : ℕ := num_pants - num_formal_pants

/-- The number of shirts that can be paired with formal pants -/
def num_shirts_for_formal : ℕ := 3

/-- Calculate the number of different outfits -/
def num_outfits : ℕ :=
  (num_casual_pants * num_shirts) + (num_formal_pants * num_shirts_for_formal)

theorem correct_outfit_count : num_outfits = 24 := by
  sorry

end NUMINAMATH_CALUDE_correct_outfit_count_l2116_211665


namespace NUMINAMATH_CALUDE_division_multiplication_relation_l2116_211671

theorem division_multiplication_relation (a b c : ℕ) (h : a / b = c) : 
  c * b = a ∧ a / c = b :=
by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_relation_l2116_211671


namespace NUMINAMATH_CALUDE_uma_income_l2116_211689

theorem uma_income (uma_income bala_income uma_expenditure bala_expenditure : ℚ)
  (income_ratio : uma_income / bala_income = 4 / 3)
  (expenditure_ratio : uma_expenditure / bala_expenditure = 3 / 2)
  (uma_savings : uma_income - uma_expenditure = 5000)
  (bala_savings : bala_income - bala_expenditure = 5000) :
  uma_income = 20000 := by
  sorry

end NUMINAMATH_CALUDE_uma_income_l2116_211689


namespace NUMINAMATH_CALUDE_no_prime_divisible_by_55_l2116_211666

theorem no_prime_divisible_by_55 : ¬ ∃ p : ℕ, Nat.Prime p ∧ 55 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_no_prime_divisible_by_55_l2116_211666


namespace NUMINAMATH_CALUDE_cos_plus_sin_value_l2116_211633

theorem cos_plus_sin_value (α : Real) (k : Real) :
  (∃ x y : Real, x * y = 1 ∧ 
    x^2 - k*x + k^2 - 3 = 0 ∧ 
    y^2 - k*y + k^2 - 3 = 0 ∧ 
    x = Real.tan α ∧ 
    y = 1 / Real.tan α) →
  3 * Real.pi < α ∧ α < 7/2 * Real.pi →
  Real.cos α + Real.sin α = -Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cos_plus_sin_value_l2116_211633


namespace NUMINAMATH_CALUDE_distributive_property_example_l2116_211643

theorem distributive_property_example :
  (3/4 + 7/12 - 5/9) * (-36) = 3/4 * (-36) + 7/12 * (-36) - 5/9 * (-36) := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_example_l2116_211643


namespace NUMINAMATH_CALUDE_simone_apple_days_l2116_211624

theorem simone_apple_days (d : ℕ) : 
  (1/2 : ℚ) * d + (1/3 : ℚ) * 15 = 13 → d = 16 := by
  sorry

end NUMINAMATH_CALUDE_simone_apple_days_l2116_211624


namespace NUMINAMATH_CALUDE_not_divide_power_plus_one_l2116_211680

theorem not_divide_power_plus_one (p q m : ℕ) : 
  Nat.Prime p → Nat.Prime q → Odd p → Odd q → p > q → m > 0 → ¬(p * q ∣ m^(p - q) + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divide_power_plus_one_l2116_211680


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l2116_211641

/-- Given three lines that intersect at the same point, prove the value of m -/
theorem intersection_of_three_lines (x y : ℝ) (m : ℝ) :
  (y = 4 * x + 2) ∧ 
  (y = -3 * x - 18) ∧ 
  (y = 2 * x + m) →
  m = -26 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l2116_211641


namespace NUMINAMATH_CALUDE_rod_length_difference_l2116_211660

theorem rod_length_difference (L₁ L₂ : ℝ) : 
  L₁ + L₂ = 33 →
  (1 - 1/3) * L₁ = (1 - 1/5) * L₂ →
  L₁ - L₂ = 3 := by
sorry

end NUMINAMATH_CALUDE_rod_length_difference_l2116_211660


namespace NUMINAMATH_CALUDE_contrapositive_is_true_l2116_211670

-- Define the original proposition
def original_proposition (x y : ℝ) : Prop :=
  (x = 2 ∧ y = 3) → (x + y = 5)

-- Define the contrapositive of the original proposition
def contrapositive (x y : ℝ) : Prop :=
  (x + y ≠ 5) → (x ≠ 2 ∨ y ≠ 3)

-- Theorem stating that the contrapositive is true
theorem contrapositive_is_true : ∀ x y : ℝ, contrapositive x y :=
by
  sorry

end NUMINAMATH_CALUDE_contrapositive_is_true_l2116_211670


namespace NUMINAMATH_CALUDE_travel_cost_theorem_l2116_211659

structure City where
  name : String

structure Triangle where
  D : City
  E : City
  F : City
  DE : ℝ
  EF : ℝ
  FD : ℝ
  right_angle_at_D : DE^2 + FD^2 = EF^2

def bus_fare_per_km : ℝ := 0.20

def plane_booking_fee (departure : City) : ℝ :=
  if departure.name = "E" then 150 else 120

def plane_fare_per_km : ℝ := 0.12

def travel_cost (t : Triangle) : ℝ :=
  t.DE * bus_fare_per_km +
  t.EF * plane_fare_per_km +
  plane_booking_fee t.E

theorem travel_cost_theorem (t : Triangle) :
  t.DE = 4000 ∧ t.EF = 4500 ∧ t.FD = 5000 →
  travel_cost t = 1490 := by
  sorry

end NUMINAMATH_CALUDE_travel_cost_theorem_l2116_211659


namespace NUMINAMATH_CALUDE_unique_quadratic_function_max_min_values_l2116_211673

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem unique_quadratic_function 
  (a b : ℝ) 
  (h1 : a ≠ 0)
  (h2 : f a b 2 = 0)
  (h3 : ∃! x, f a b x = x) :
  ∀ x, f a b x = -1/2 * x^2 + x := by
sorry

-- For the second part of the question
theorem max_min_values
  (h : ∀ x, f 1 (-2) x = x^2 - 2*x) :
  (∀ x ∈ [-1, 2], f 1 (-2) x ≤ 3) ∧
  (∀ x ∈ [-1, 2], f 1 (-2) x ≥ -1) ∧
  (∃ x ∈ [-1, 2], f 1 (-2) x = 3) ∧
  (∃ x ∈ [-1, 2], f 1 (-2) x = -1) := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_function_max_min_values_l2116_211673


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2116_211663

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x > 0, x^2 - 5*x + 6 = 0) ↔ (∀ x > 0, x^2 - 5*x + 6 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2116_211663


namespace NUMINAMATH_CALUDE_marks_theater_cost_l2116_211615

/-- The cost of Mark's theater visits over a given number of weeks -/
def theater_cost (weeks : ℕ) (hours_per_visit : ℕ) (price_per_hour : ℕ) : ℕ :=
  weeks * hours_per_visit * price_per_hour

/-- Theorem: Mark's theater visits cost $90 over 6 weeks -/
theorem marks_theater_cost :
  theater_cost 6 3 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_marks_theater_cost_l2116_211615


namespace NUMINAMATH_CALUDE_log_stack_sum_l2116_211678

theorem log_stack_sum : ∀ (a l : ℕ) (d : ℤ),
  a = 15 ∧ l = 4 ∧ d = -1 →
  ∃ n : ℕ, n > 0 ∧ l = a + (n - 1 : ℤ) * d ∧
  (n : ℤ) * (a + l) / 2 = 114 :=
by sorry

end NUMINAMATH_CALUDE_log_stack_sum_l2116_211678


namespace NUMINAMATH_CALUDE_a_in_N_necessary_not_sufficient_for_a_in_M_l2116_211661

def M : Set ℝ := {x | 0 < x ∧ x < 1}
def N : Set ℝ := {x | -2 < x ∧ x < 1}

theorem a_in_N_necessary_not_sufficient_for_a_in_M :
  (∀ a, a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by sorry

end NUMINAMATH_CALUDE_a_in_N_necessary_not_sufficient_for_a_in_M_l2116_211661


namespace NUMINAMATH_CALUDE_common_chord_equation_l2116_211684

/-- The equation of the line where the common chord of two circles lies -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 - 2*x - 8 = 0) ∧ (x^2 + y^2 + 2*x - 4*y - 4 = 0) →
  (x - y + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2116_211684


namespace NUMINAMATH_CALUDE_specific_pairing_probability_l2116_211668

/-- The probability of a specific pairing in a classroom with random pairings -/
theorem specific_pairing_probability 
  (n : ℕ) -- Total number of students
  (h : n = 32) -- Given number of students in the classroom
  : (1 : ℚ) / (n - 1 : ℚ) = 1 / 31 := by
  sorry

#check specific_pairing_probability

end NUMINAMATH_CALUDE_specific_pairing_probability_l2116_211668


namespace NUMINAMATH_CALUDE_positive_A_value_l2116_211617

/-- The relation # is defined as A # B = A^2 + B^2 -/
def hash (A B : ℝ) : ℝ := A^2 + B^2

/-- Given A # 7 = 194, prove that the positive value of A is √145 -/
theorem positive_A_value (h : hash A 7 = 194) : A = Real.sqrt 145 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l2116_211617


namespace NUMINAMATH_CALUDE_equation_decomposition_l2116_211656

-- Define the original equation
def original_equation (y z : ℝ) : Prop :=
  z^4 - 6*y^4 = 3*z^2 - 2

-- Define the hyperbola equation
def hyperbola_equation (y z : ℝ) : Prop :=
  z^2 - 3*y^2 = 2

-- Define the ellipse equation
def ellipse_equation (y z : ℝ) : Prop :=
  z^2 - 2*y^2 = 1

-- Theorem stating that the original equation can be decomposed into a hyperbola and an ellipse
theorem equation_decomposition :
  ∀ y z : ℝ, original_equation y z ↔ (hyperbola_equation y z ∨ ellipse_equation y z) :=
by sorry


end NUMINAMATH_CALUDE_equation_decomposition_l2116_211656


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l2116_211694

theorem arithmetic_geometric_mean_ratio (a b m : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (hm : (a + b) / 2 = m * Real.sqrt (a * b)) : 
  a / b = (m + Real.sqrt (m^2 + 1)) / (m - Real.sqrt (m^2 - 1)) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l2116_211694


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2116_211691

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 2 - 3 * Complex.I) * 
               (Real.sqrt 5 + 5 * Complex.I) * 
               (2 - 2 * Complex.I)) = 18 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2116_211691


namespace NUMINAMATH_CALUDE_odd_function_with_minimum_l2116_211604

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem odd_function_with_minimum (a b c d : ℝ) :
  (∀ x, f a b c d x = -f a b c d (-x)) →  -- f is an odd function
  (∀ x, f a b c d x ≥ f a b c d (-1)) →   -- f(-1) is the minimum value
  f a b c d (-1) = -1 →                   -- f(-1) = -1
  (∀ x, f a b c d x = -x^3 + x) :=        -- Conclusion: f(x) = -x³ + x
by
  sorry


end NUMINAMATH_CALUDE_odd_function_with_minimum_l2116_211604


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_5pi_18_l2116_211693

theorem sin_2alpha_plus_5pi_18 (α : ℝ) (h : Real.sin (π / 9 - α) = 1 / 3) : 
  Real.sin (2 * α + 5 * π / 18) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_5pi_18_l2116_211693


namespace NUMINAMATH_CALUDE_min_value_cyclic_fraction_l2116_211650

theorem min_value_cyclic_fraction (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a / b + b / c + c / d + d / a ≥ 4 ∧ 
  (a / b + b / c + c / d + d / a = 4 ↔ a = b ∧ b = c ∧ c = d) := by
  sorry

end NUMINAMATH_CALUDE_min_value_cyclic_fraction_l2116_211650


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_four_l2116_211690

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum is 4 or less -/
def outcomes_sum_4_or_less : ℕ := 6

/-- The probability of an event occurring is the number of favorable outcomes
    divided by the total number of possible outcomes -/
theorem prob_sum_greater_than_four :
  (1 - (outcomes_sum_4_or_less : ℚ) / total_outcomes) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_four_l2116_211690


namespace NUMINAMATH_CALUDE_segment_ratio_vector_coefficients_l2116_211620

-- Define the vector type
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points C, D, and Q
variable (C D Q : V)

-- Define the condition that Q is on the line segment CD with the given ratio
def on_segment_with_ratio (C D Q : V) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D ∧ t = 5 / 8

-- Theorem statement
theorem segment_ratio_vector_coefficients
  (h : on_segment_with_ratio C D Q) :
  ∃ (s v : ℝ), Q = s • C + v • D ∧ s = 5/8 ∧ v = 3/8 :=
sorry

end NUMINAMATH_CALUDE_segment_ratio_vector_coefficients_l2116_211620


namespace NUMINAMATH_CALUDE_percentage_unsold_bags_l2116_211628

/-- Given the initial stock and daily sales of bags in a bookshop,
    prove that the percentage of unsold bags is 25%. -/
theorem percentage_unsold_bags
  (initial_stock : ℕ)
  (monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales : ℕ)
  (h_initial : initial_stock = 600)
  (h_monday : monday_sales = 25)
  (h_tuesday : tuesday_sales = 70)
  (h_wednesday : wednesday_sales = 100)
  (h_thursday : thursday_sales = 110)
  (h_friday : friday_sales = 145) :
  (initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) / initial_stock * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_unsold_bags_l2116_211628


namespace NUMINAMATH_CALUDE_water_level_rise_l2116_211658

/-- Calculates the rise in water level when a cube is immersed in a rectangular vessel -/
theorem water_level_rise
  (cube_edge : ℝ)
  (vessel_length : ℝ)
  (vessel_width : ℝ)
  (h_cube_edge : cube_edge = 10)
  (h_vessel_length : vessel_length = 20)
  (h_vessel_width : vessel_width = 15) :
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_water_level_rise_l2116_211658


namespace NUMINAMATH_CALUDE_calculation_21_implies_72_l2116_211602

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  h_tens : tens ≥ 1 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The calculation process described in the problem -/
def calculation (n : TwoDigitNumber) : Nat :=
  2 * (5 * n.units - 3) + n.tens

/-- Theorem stating that if the calculation result is 21, the original number is 72 -/
theorem calculation_21_implies_72 (n : TwoDigitNumber) :
  calculation n = 21 → n.tens = 7 ∧ n.units = 2 := by
  sorry

#eval calculation ⟨7, 2, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_calculation_21_implies_72_l2116_211602


namespace NUMINAMATH_CALUDE_total_jars_is_24_l2116_211653

/-- Represents the number of each type of jar -/
def num_each_jar : ℕ := 8

/-- Represents the total volume of water in gallons -/
def total_water : ℕ := 14

/-- Represents the volume of water in quarts held by all quart jars -/
def quart_jars_volume : ℕ := num_each_jar

/-- Represents the volume of water in quarts held by all half-gallon jars -/
def half_gallon_jars_volume : ℕ := 2 * num_each_jar

/-- Represents the volume of water in quarts held by all one-gallon jars -/
def gallon_jars_volume : ℕ := 4 * num_each_jar

/-- Theorem stating that the total number of water-filled jars is 24 -/
theorem total_jars_is_24 : 
  quart_jars_volume + half_gallon_jars_volume + gallon_jars_volume = total_water * 4 ∧
  3 * num_each_jar = 24 := by
  sorry

#check total_jars_is_24

end NUMINAMATH_CALUDE_total_jars_is_24_l2116_211653


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2116_211654

/-- Two 2D vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (-1, 2*m + 1)
  parallel a b → m = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2116_211654


namespace NUMINAMATH_CALUDE_geometric_subsequence_ratio_l2116_211644

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arith : ∀ n, a (n + 1) = a n + d
  h_d_nonzero : d ≠ 0

/-- The property that a_1, a_3, and a_7 form a geometric sequence -/
def IsGeometricSubsequence (seq : ArithmeticSequence) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ seq.a 3 = seq.a 1 * q ∧ seq.a 7 = seq.a 3 * q

/-- The theorem stating that the common ratio of the geometric subsequence is 2 -/
theorem geometric_subsequence_ratio (seq : ArithmeticSequence) 
  (h_geom : IsGeometricSubsequence seq) : 
  ∃ q : ℝ, q = 2 ∧ seq.a 3 = seq.a 1 * q ∧ seq.a 7 = seq.a 3 * q := by
  sorry

end NUMINAMATH_CALUDE_geometric_subsequence_ratio_l2116_211644


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2116_211618

/-- Given a geometric sequence {a_n} with a_2 = 2 and S_3 = 8, prove S_5 / a_3 = 11 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1)  -- Geometric sequence condition
  → a 2 = 2                           -- Given condition
  → S 3 = 8                           -- Given condition
  → S 5 / a 3 = 11 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2116_211618
